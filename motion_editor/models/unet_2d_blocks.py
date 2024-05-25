# Adapted from https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/unet_2d_blocks.py
from typing import Optional

import torch
from torch import nn
from einops import rearrange, repeat
import torch.nn.functional as F
from diffusers.models.attention import FeedForward
from .attention_2d import Transformer2DModel
from .resnet_2d import Downsample2D, ResnetBlock2D, Upsample2D

from diffusers.utils.import_utils import is_xformers_available

if is_xformers_available():
    import xformers
    import xformers.ops
else:
    print(1/0)
    xformers = None

class AdapterCrossAttention(nn.Module):
    def __init__(
            self,
            query_dim: int,
            cross_attention_dim: Optional[int] = None,
            heads: int = 8,
            dim_head: int = 64,
            dropout: float = 0.0,
            bias=False,
            upcast_attention: bool = False,
            upcast_softmax: bool = False,
            added_kv_proj_dim: Optional[int] = None,
            norm_num_groups: Optional[int] = None,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        cross_attention_dim = cross_attention_dim if cross_attention_dim is not None else query_dim
        self.upcast_attention = upcast_attention
        self.upcast_softmax = upcast_softmax

        self.scale = dim_head ** -0.5

        self.heads = heads
        # for slice_size > 0 the attention score computation
        # is split across the batch axis to save memory
        # You can set slice_size with `set_attention_slice`
        self.sliceable_head_dim = heads
        self._slice_size = None
        self._use_memory_efficient_attention_xformers = True
        self.added_kv_proj_dim = added_kv_proj_dim

        if norm_num_groups is not None:
            self.group_norm = nn.GroupNorm(num_channels=inner_dim, num_groups=norm_num_groups, eps=1e-5, affine=True)
        else:
            self.group_norm = None

        self.to_q = nn.Linear(query_dim, inner_dim, bias=bias)
        self.to_k = nn.Linear(cross_attention_dim, inner_dim, bias=bias)
        self.to_v = nn.Linear(cross_attention_dim, inner_dim, bias=bias)

        if self.added_kv_proj_dim is not None:
            self.add_k_proj = nn.Linear(added_kv_proj_dim, cross_attention_dim)
            self.add_v_proj = nn.Linear(added_kv_proj_dim, cross_attention_dim)

        self.to_out = nn.ModuleList([])
        self.to_out.append(nn.Linear(inner_dim, query_dim))
        self.to_out.append(nn.Dropout(dropout))

    def reshape_heads_to_batch_dim(self, tensor):
        batch_size, seq_len, dim = tensor.shape
        head_size = self.heads
        tensor = tensor.reshape(batch_size, seq_len, head_size, dim // head_size)
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size * head_size, seq_len, dim // head_size)
        return tensor

    def reshape_batch_dim_to_heads(self, tensor):
        batch_size, seq_len, dim = tensor.shape
        head_size = self.heads
        tensor = tensor.reshape(batch_size // head_size, head_size, seq_len, dim)
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size // head_size, seq_len, dim * head_size)
        return tensor

    def set_attention_slice(self, slice_size):
        if slice_size is not None and slice_size > self.sliceable_head_dim:
            raise ValueError(f"slice_size {slice_size} has to be smaller or equal to {self.sliceable_head_dim}.")

        self._slice_size = slice_size

    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, normal_infer=False):
        batch_size, sequence_length, _ = hidden_states.shape

        encoder_hidden_states = encoder_hidden_states

        if self.group_norm is not None:
            hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = self.to_q(hidden_states)
        dim = query.shape[-1]
        query = self.reshape_heads_to_batch_dim(query)

        if self.added_kv_proj_dim is not None:
            key = self.to_k(hidden_states)
            value = self.to_v(hidden_states)
            encoder_hidden_states_key_proj = self.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = self.add_v_proj(encoder_hidden_states)

            key = self.reshape_heads_to_batch_dim(key)
            value = self.reshape_heads_to_batch_dim(value)
            encoder_hidden_states_key_proj = self.reshape_heads_to_batch_dim(encoder_hidden_states_key_proj)
            encoder_hidden_states_value_proj = self.reshape_heads_to_batch_dim(encoder_hidden_states_value_proj)

            key = torch.concat([encoder_hidden_states_key_proj, key], dim=1)
            value = torch.concat([encoder_hidden_states_value_proj, value], dim=1)
        else:
            encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
            key = self.to_k(encoder_hidden_states)
            value = self.to_v(encoder_hidden_states)

            key = self.reshape_heads_to_batch_dim(key)
            value = self.reshape_heads_to_batch_dim(value)

        if attention_mask is not None:
            if attention_mask.shape[-1] != query.shape[1]:
                target_length = query.shape[1]
                attention_mask = F.pad(attention_mask, (0, target_length), value=0.0)
                attention_mask = attention_mask.repeat_interleave(self.heads, dim=0)

        # attention, what we cannot get enough of
        if self._use_memory_efficient_attention_xformers:
            hidden_states = self._memory_efficient_attention_xformers(query, key, value, attention_mask)
            # Some versions of xformers return output in fp32, cast it back to the dtype of the input
            hidden_states = hidden_states.to(query.dtype)
        else:
            if self._slice_size is None or query.shape[0] // self._slice_size == 1:
                hidden_states = self._attention(query, key, value, attention_mask)
            else:
                hidden_states = self._sliced_attention(query, key, value, sequence_length, dim, attention_mask)

        # linear proj
        hidden_states = self.to_out[0](hidden_states)

        # dropout
        hidden_states = self.to_out[1](hidden_states)
        return hidden_states

    def _attention(self, query, key, value, attention_mask=None):
        if self.upcast_attention:
            query = query.float()
            key = key.float()

        attention_scores = torch.baddbmm(
            torch.empty(query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device),
            query,
            key.transpose(-1, -2),
            beta=0,
            alpha=self.scale,
        )

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        if self.upcast_softmax:
            attention_scores = attention_scores.float()

        attention_probs = attention_scores.softmax(dim=-1)

        # cast back to the original dtype
        attention_probs = attention_probs.to(value.dtype)

        # compute attention output
        hidden_states = torch.bmm(attention_probs, value)

        # reshape hidden_states
        hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
        return hidden_states

    def _sliced_attention(self, query, key, value, sequence_length, dim, attention_mask):
        batch_size_attention = query.shape[0]
        hidden_states = torch.zeros(
            (batch_size_attention, sequence_length, dim // self.heads), device=query.device, dtype=query.dtype
        )
        slice_size = self._slice_size if self._slice_size is not None else hidden_states.shape[0]
        for i in range(hidden_states.shape[0] // slice_size):
            start_idx = i * slice_size
            end_idx = (i + 1) * slice_size

            query_slice = query[start_idx:end_idx]
            key_slice = key[start_idx:end_idx]

            if self.upcast_attention:
                query_slice = query_slice.float()
                key_slice = key_slice.float()

            attn_slice = torch.baddbmm(
                torch.empty(slice_size, query.shape[1], key.shape[1], dtype=query_slice.dtype, device=query.device),
                query_slice,
                key_slice.transpose(-1, -2),
                beta=0,
                alpha=self.scale,
            )

            if attention_mask is not None:
                attn_slice = attn_slice + attention_mask[start_idx:end_idx]

            if self.upcast_softmax:
                attn_slice = attn_slice.float()

            attn_slice = attn_slice.softmax(dim=-1)

            # cast back to the original dtype
            attn_slice = attn_slice.to(value.dtype)
            attn_slice = torch.bmm(attn_slice, value[start_idx:end_idx])

            hidden_states[start_idx:end_idx] = attn_slice

        # reshape hidden_states
        hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
        return hidden_states

    def _memory_efficient_attention_xformers(self, query, key, value, attention_mask):
        # TODO attention_mask
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()
        hidden_states = xformers.ops.memory_efficient_attention(query, key, value, attn_bias=attention_mask)
        hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
        return hidden_states

def conv_nd(dims, in_channels, out_channels, kernel_size, **kwargs):
    if dims == 1:
        return nn.Conv1d(in_channels, out_channels, kernel_size, **kwargs)
    elif dims == 2:
        return nn.Conv2d(in_channels, out_channels, kernel_size, **kwargs)
    elif dims == 3:
        if isinstance(kernel_size, int):
            kernel_size = (1, *((kernel_size,) * 2))
        if 'stride' in kwargs.keys():
            if isinstance(kwargs['stride'], int):
                kwargs['stride'] = (1, *((kwargs['stride'],) * 2))
        if 'padding' in kwargs.keys():
            if isinstance(kwargs['padding'], int):
                kwargs['padding'] = (0, *((kwargs['padding'],) * 2))
        return nn.Conv3d(in_channels, out_channels, kernel_size, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def avg_pool_nd(dims, *args, **kwargs):
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")

class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, 3, stride=stride, padding=padding
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


class ResnetBlock(nn.Module):
    def __init__(self, in_c, out_c, down, ksize=3, sk=False, use_conv=True, num_attention_heads=16, cross_prompt_attention_dim=768):
        super().__init__()
        ps = ksize // 2
        if in_c != out_c or sk == False:
            self.in_conv = nn.Conv2d(in_c, out_c, ksize, 1, ps)
        else:
            # print('n_in')
            self.in_conv = None
        self.block1 = nn.Conv2d(out_c, out_c, 3, 1, 1)
        self.act = nn.ReLU()
        self.block2 = nn.Conv2d(out_c, out_c, ksize, 1, ps)
        if sk == False:
            self.skep = nn.Conv2d(in_c, out_c, ksize, 1, ps)
        else:
            self.skep = None

        self.down = down
        if self.down == True:
            self.down_opt = Downsample(in_c, use_conv=use_conv)

        # Temporal Convolution
        self.conv1 = nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1)

        self.attn_spatial = AdapterCrossAttention(
            query_dim=out_c,
            heads=num_attention_heads,
            dim_head=out_c // num_attention_heads,
            dropout=0.0,
            bias=False,
            cross_attention_dim=out_c,
            upcast_attention=False,
        )
        self.ff = FeedForward(out_c, dropout=0.0, activation_fn="geglu")
        self.ff_norm = nn.LayerNorm(out_c)
        self.norm_temp = nn.LayerNorm(out_c)

        # Cross Attention between source video feature and pose feature
        self.cross_pose_attn = AdapterCrossAttention(
            query_dim=out_c,
            cross_attention_dim=out_c,
            heads=num_attention_heads,
            dim_head=out_c // num_attention_heads,
            dropout=0.0,
            bias=False,
            upcast_attention=False)
        self.cross_pose_norm = nn.LayerNorm(out_c)
        self.cross_pose_attn = zero_module(self.cross_pose_attn)

    def forward(self, x, encoder_hidden_states=None, source_hidden_states=None):
        if self.down == True:
            x = self.down_opt(x)
        if self.in_conv is not None:  # edit
            x = self.in_conv(x)

        h = self.block1(x)
        h = h + self.conv1(h)
        h = self.act(h)
        h = self.block2(h)
        h = h + self.conv2(h)

        if self.skep is not None:
            h = h + self.skep(x)
        else:
            h = h + x

        # Temporal-Attention
        batch, c_dim, h_dim, w_dim = h.size()
        h = rearrange(h, "b c h w -> b (h w) c", b=batch, c=c_dim, h=h_dim, w=w_dim)
        norm_hidden_states = (self.norm_temp(h))

        hidden_states = self.attn_spatial(norm_hidden_states, normal_infer=False) + h

        # Cross Attention between source features and pose features
        source_batch, source_c_dim, source_h_dim, source_w_dim = source_hidden_states.size()
        source_hidden_states = rearrange(source_hidden_states, "b c h w -> b (h w) c", b=source_batch, c=source_c_dim, h=source_h_dim, w=source_w_dim)
        hidden_states = self.cross_pose_norm(hidden_states)
        hidden_states = self.cross_pose_attn(hidden_states, encoder_hidden_states=source_hidden_states) + hidden_states
        hidden_states = self.ff(self.ff_norm(hidden_states)) + hidden_states

        hidden_states = rearrange(hidden_states, "b (h w) c -> b c h w", h=h_dim, w=w_dim)

        return hidden_states


class Detached_Adapter(nn.Module):
    def __init__(self, channels=[320, 640, 1280, 1280], nums_rb=3, cin=64, ksize=3, sk=False, use_conv=True, idx=None):
        super().__init__()
        self.idx = idx
        if idx == 0:
            self.unshuffle = nn.PixelUnshuffle(8)
            self.conv_in = nn.Conv2d(cin, channels[0], 3, 1, 1)
        self.channels = channels
        self.nums_rb = nums_rb
        self.body = []
        for j in range(nums_rb):
            if (self.idx != 0) and (j == 0):
                self.body.append(ResnetBlock(channels[self.idx-1], channels[self.idx], down=True, ksize=ksize, sk=sk, use_conv=use_conv))
            else:
                self.body.append(ResnetBlock(channels[self.idx], channels[self.idx], down=False, ksize=ksize, sk=sk, use_conv=use_conv))
        self.body = nn.ModuleList(self.body)

    def forward(self, x, encoder_hidden_states=None, source_hidden_states=None):
        if self.idx == 0:
            b, c, h, w = x.shape
            x = self.unshuffle(x)
            x = self.conv_in(x)
        else:
            b, c, h, w = x.shape
        for j in range(self.nums_rb):
            x = self.body[j](x, encoder_hidden_states=encoder_hidden_states, source_hidden_states=source_hidden_states)
        return x

def get_down_block(
    down_block_type,
    num_layers,
    in_channels,
    out_channels,
    temb_channels,
    add_downsample,
    resnet_eps,
    resnet_act_fn,
    attn_num_head_channels,
    resnet_groups=None,
    cross_attention_dim=None,
    downsample_padding=None,
    dual_cross_attention=False,
    use_linear_projection=False,
    only_cross_attention=False,
    upcast_attention=False,
    resnet_time_scale_shift="default",
    use_sc_attn=False,
    use_st_attn=False,
    detached_idx=None,
):
    down_block_type = down_block_type[7:] if down_block_type.startswith("UNetRes") else down_block_type
    if down_block_type == "DownBlock2D":
        return DownBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            add_downsample=add_downsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            downsample_padding=downsample_padding,
            resnet_time_scale_shift=resnet_time_scale_shift,
            detached_idx=detached_idx,
        )
    elif down_block_type == "CrossAttnDownBlock2D":
        if cross_attention_dim is None:
            raise ValueError("cross_attention_dim must be specified for CrossAttnDownBlock2D")
        return CrossAttnDownBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            add_downsample=add_downsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            downsample_padding=downsample_padding,
            cross_attention_dim=cross_attention_dim,
            attn_num_head_channels=attn_num_head_channels,
            dual_cross_attention=dual_cross_attention,
            use_linear_projection=use_linear_projection,
            only_cross_attention=only_cross_attention,
            upcast_attention=upcast_attention,
            resnet_time_scale_shift=resnet_time_scale_shift,
            use_sc_attn=use_sc_attn,
            use_st_attn=use_st_attn,
            detached_idx=detached_idx,
        )
    raise ValueError(f"{down_block_type} does not exist.")


def get_up_block(
    up_block_type,
    num_layers,
    in_channels,
    out_channels,
    prev_output_channel,
    temb_channels,
    add_upsample,
    resnet_eps,
    resnet_act_fn,
    attn_num_head_channels,
    resnet_groups=None,
    cross_attention_dim=None,
    dual_cross_attention=False,
    use_linear_projection=False,
    only_cross_attention=False,
    upcast_attention=False,
    resnet_time_scale_shift="default",
    use_sc_attn=False,
    use_st_attn=False,
):
    up_block_type = up_block_type[7:] if up_block_type.startswith("UNetRes") else up_block_type
    if up_block_type == "UpBlock2D":
        return UpBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            prev_output_channel=prev_output_channel,
            temb_channels=temb_channels,
            add_upsample=add_upsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            resnet_time_scale_shift=resnet_time_scale_shift,
        )
    elif up_block_type == "CrossAttnUpBlock2D":
        if cross_attention_dim is None:
            raise ValueError("cross_attention_dim must be specified for CrossAttnUpBlock2D")
        return CrossAttnUpBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            prev_output_channel=prev_output_channel,
            temb_channels=temb_channels,
            add_upsample=add_upsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            cross_attention_dim=cross_attention_dim,
            attn_num_head_channels=attn_num_head_channels,
            dual_cross_attention=dual_cross_attention,
            use_linear_projection=use_linear_projection,
            only_cross_attention=only_cross_attention,
            upcast_attention=upcast_attention,
            resnet_time_scale_shift=resnet_time_scale_shift,
            use_sc_attn=use_sc_attn,
            use_st_attn=use_st_attn,
        )
    raise ValueError(f"{up_block_type} does not exist.")


class UNetMidBlock2DCrossAttn(nn.Module):
    def __init__(
        self,
        in_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        attn_num_head_channels=1,
        output_scale_factor=1.0,
        cross_attention_dim=1280,
        dual_cross_attention=False,
        use_linear_projection=False,
        upcast_attention=False,
        use_sc_attn=False,
        use_st_attn=False,
    ):
        super().__init__()

        self.has_cross_attention = True
        self.attn_num_head_channels = attn_num_head_channels
        resnet_groups = resnet_groups if resnet_groups is not None else min(in_channels // 4, 32)

        # there is always at least one resnet
        resnets = [
            ResnetBlock2D(
                in_channels=in_channels,
                out_channels=in_channels,
                temb_channels=temb_channels,
                eps=resnet_eps,
                groups=resnet_groups,
                dropout=dropout,
                time_embedding_norm=resnet_time_scale_shift,
                non_linearity=resnet_act_fn,
                output_scale_factor=output_scale_factor,
                pre_norm=resnet_pre_norm,
            )
        ]
        attentions = []

        for _ in range(num_layers):
            if dual_cross_attention:
                raise NotImplementedError
            attentions.append(
                Transformer2DModel(
                    attn_num_head_channels,
                    in_channels // attn_num_head_channels,
                    in_channels=in_channels,
                    num_layers=1,
                    cross_attention_dim=cross_attention_dim,
                    norm_num_groups=resnet_groups,
                    use_linear_projection=use_linear_projection,
                    upcast_attention=upcast_attention,
                    use_sc_attn=use_sc_attn,
                    use_st_attn=True if (use_st_attn and _ == 0) else False,
                )
            )
            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

    def forward(self, hidden_states, temb=None, encoder_hidden_states=None, attention_mask=None, normal_infer=False, temb_aux=None, masks=None, source_masks=None, target_masks=None, rectangle_source_masks=None,):
        hidden_states = self.resnets[0](hidden_states, temb, temb_aux=temb_aux, masks=masks)
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            hidden_states = attn(hidden_states, encoder_hidden_states=encoder_hidden_states, normal_infer=normal_infer, source_masks=None, target_masks=None, rectangle_source_masks=None).sample
            hidden_states = resnet(hidden_states, temb, temb_aux=temb_aux, masks=masks, source_masks=source_masks, target_masks=target_masks)

        return hidden_states


class CrossAttnDownBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        attn_num_head_channels=1,
        cross_attention_dim=1280,
        output_scale_factor=1.0,
        downsample_padding=1,
        add_downsample=True,
        dual_cross_attention=False,
        use_linear_projection=False,
        only_cross_attention=False,
        upcast_attention=False,
        use_sc_attn=False,
        use_st_attn=False,
        detached_idx=None,
    ):
        super().__init__()
        resnets = []
        attentions = []

        self.has_cross_attention = True
        self.attn_num_head_channels = attn_num_head_channels

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )
            if dual_cross_attention:
                raise NotImplementedError
            attentions.append(
                Transformer2DModel(
                    attn_num_head_channels,
                    out_channels // attn_num_head_channels,
                    in_channels=out_channels,
                    num_layers=1,
                    cross_attention_dim=cross_attention_dim,
                    norm_num_groups=resnet_groups,
                    use_linear_projection=use_linear_projection,
                    only_cross_attention=only_cross_attention,
                    upcast_attention=upcast_attention,
                    use_sc_attn=use_sc_attn,
                    use_st_attn=True if (use_st_attn and i == 0) else False,
                )
            )
        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    Downsample2D(
                        out_channels, use_conv=True, out_channels=out_channels, padding=downsample_padding, name="op"
                    )
                ]
            )
        else:
            self.downsamplers = None

        self.gradient_checkpointing = False

        self.detached_idx = detached_idx
        # self.cross_pose_attn = Detached_Adapter(cin=int(3 * 64), channels=[320, 640, 1280, 1280][:4], nums_rb=2, ksize=1, sk=True, use_conv=False, idx=detached_idx)

    def forward(self, hidden_states, temb=None, encoder_hidden_states=None, attention_mask=None, normal_infer=False, skeleton=None, temb_aux=None, masks=None, source_masks=None, target_masks=None, rectangle_source_masks=None):
        output_states = ()

        idx = 1
        for resnet, attn in zip(self.resnets, self.attentions):
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None, normal_infer=False):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict, normal_infer=normal_infer)
                        else:
                            return module(*inputs)

                    return custom_forward

                hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(resnet), hidden_states, temb, temb_aux, masks, source_masks, target_masks)
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(attn, return_dict=False, normal_infer=normal_infer),
                    hidden_states,
                    encoder_hidden_states,
                    source_masks,
                    target_masks,
                    rectangle_source_masks,
                )[0]

                if (idx + 1) % 3 == 0 and skeleton is not None:
                    if hidden_states.size()[0] == 4:
                        if self.detached_idx == 0:
                            target_skeleton_1 = torch.unsqueeze(skeleton[1], dim=0)
                            num_frames = target_skeleton_1.size()[1]
                            target_skeleton_1 = rearrange(target_skeleton_1, "b f c h w -> (b f) c h w")
                            target_skeleton_2 = torch.unsqueeze(skeleton[3], dim=0)
                            target_skeleton_2 = rearrange(target_skeleton_2, "b f c h w -> (b f) c h w")
                            target_skeleton = torch.cat([target_skeleton_1, target_skeleton_2], dim=0)
                            target_hidden_states_1 = torch.unsqueeze(hidden_states[1], dim=0)
                            target_hidden_states_1 = rearrange(target_hidden_states_1, "b c f h w -> (b f) c h w")
                            target_hidden_states_2 = torch.unsqueeze(hidden_states[3], dim=0)
                            target_hidden_states_2 = rearrange(target_hidden_states_2, "b c f h w -> (b f) c h w")
                            target_hidden_states = torch.cat([target_hidden_states_1, target_hidden_states_2], dim=0)
                        else:
                            target_skeleton_1 = torch.unsqueeze(skeleton[1], dim=0)
                            num_frames = target_skeleton_1.size()[2]
                            target_skeleton_1 = rearrange(target_skeleton_1, "b c f h w -> (b f) c h w")
                            target_skeleton_2 = torch.unsqueeze(skeleton[3], dim=0)
                            target_skeleton_2 = rearrange(target_skeleton_2, "b c f h w -> (b f) c h w")
                            target_skeleton = torch.cat([target_skeleton_1, target_skeleton_2], dim=0)
                            target_hidden_states_1 = torch.unsqueeze(hidden_states[1], dim=0)
                            target_hidden_states_1 = rearrange(target_hidden_states_1, "b c f h w -> (b f) c h w")
                            target_hidden_states_2 = torch.unsqueeze(hidden_states[3], dim=0)
                            target_hidden_states_2 = rearrange(target_hidden_states_2, "b c f h w -> (b f) c h w")
                            target_hidden_states = torch.cat([target_hidden_states_1, target_hidden_states_2], dim=0)
                        target_skeleton = self.cross_pose_attn(target_skeleton, source_hidden_states=target_hidden_states)
                        batch_frames, c_dim, h_dim, w_dim = target_skeleton.size()
                        target_skeleton = rearrange(target_skeleton, "(b f) c h w -> b c f h w", b=batch_frames//num_frames, f=num_frames, c=c_dim, h=h_dim, w=w_dim)
                        source_skeleton = torch.zeros_like(target_skeleton)
                        skeleton = torch.cat([torch.unsqueeze(source_skeleton[0], dim=0), torch.unsqueeze(target_skeleton[0], dim=0), torch.unsqueeze(source_skeleton[1], dim=0), torch.unsqueeze(target_skeleton[1], dim=0)], dim=0)
                    elif hidden_states.size()[0] == 2:
                        print(1/0)
                        skeleton = self.cross_pose_attn(target_skeleton, source_hidden_states=target_hidden_states)
                    else:
                        source_hidden_states = hidden_states
                        if self.detached_idx == 0:
                            batch, num_frames, c_dim, h_dim, w_dim = skeleton.size()
                            skeleton = rearrange(skeleton, "b f c h w -> (b f) c h w")
                            source_hidden_states = rearrange(source_hidden_states, "b c f h w -> (b f) c h w")
                        else:
                            batch, c_dim, num_frames, h_dim, w_dim = skeleton.size()
                            skeleton = rearrange(skeleton, "b c f h w -> (b f) c h w")
                            source_hidden_states = rearrange(source_hidden_states, "b c f h w -> (b f) c h w")
                        skeleton = self.cross_pose_attn(skeleton, source_hidden_states=source_hidden_states)
                        batch_frames, c_dim, h_dim, w_dim = skeleton.size()
                        skeleton = rearrange(skeleton, "(b f) c h w -> b c f h w", b=batch_frames//num_frames, f=num_frames, c=c_dim, h=h_dim, w=w_dim)
                    hidden_states = hidden_states + skeleton

                idx = idx + 1



            else:
                hidden_states = resnet(hidden_states, temb, temb_aux=temb_aux, masks=masks, source_masks=source_masks, target_masks=target_masks)
                hidden_states = attn(hidden_states, encoder_hidden_states=encoder_hidden_states, normal_infer=normal_infer, source_masks=source_masks, target_masks=target_masks, rectangle_source_masks=rectangle_source_masks).sample

                if (idx + 1) % 3 == 0 and skeleton is not None:
                    print("--------------------This is skeleton adapter in the CrossAttnDownBlock3D---------------------------")
                    print(self.detached_idx)
                    print(idx)
                    print(skeleton.size())
                    # print(encoder_hidden_states.size())
                    print(hidden_states.size())
                    print("----------------------------------------------------------------")
                    if hidden_states.size()[0] == 4:
                        if self.detached_idx == 0:
                            target_skeleton_1 = torch.unsqueeze(skeleton[1], dim=0)
                            num_frames = target_skeleton_1.size()[1]
                            target_skeleton_1 = rearrange(target_skeleton_1, "b f c h w -> (b f) c h w")
                            target_skeleton_2 = torch.unsqueeze(skeleton[3], dim=0)
                            target_skeleton_2 = rearrange(target_skeleton_2, "b f c h w -> (b f) c h w")
                            target_skeleton = torch.cat([target_skeleton_1, target_skeleton_2], dim=0)
                            target_hidden_states_1 = torch.unsqueeze(hidden_states[1], dim=0)
                            target_hidden_states_1 = rearrange(target_hidden_states_1, "b c f h w -> (b f) c h w")
                            target_hidden_states_2 = torch.unsqueeze(hidden_states[3], dim=0)
                            target_hidden_states_2 = rearrange(target_hidden_states_2, "b c f h w -> (b f) c h w")
                            target_hidden_states = torch.cat([target_hidden_states_1, target_hidden_states_2], dim=0)
                        else:
                            target_skeleton_1 = torch.unsqueeze(skeleton[1], dim=0)
                            num_frames = target_skeleton_1.size()[2]
                            target_skeleton_1 = rearrange(target_skeleton_1, "b c f h w -> (b f) c h w")
                            target_skeleton_2 = torch.unsqueeze(skeleton[3], dim=0)
                            target_skeleton_2 = rearrange(target_skeleton_2, "b c f h w -> (b f) c h w")
                            target_skeleton = torch.cat([target_skeleton_1, target_skeleton_2], dim=0)
                            target_hidden_states_1 = torch.unsqueeze(hidden_states[1], dim=0)
                            target_hidden_states_1 = rearrange(target_hidden_states_1, "b c f h w -> (b f) c h w")
                            target_hidden_states_2 = torch.unsqueeze(hidden_states[3], dim=0)
                            target_hidden_states_2 = rearrange(target_hidden_states_2, "b c f h w -> (b f) c h w")
                            target_hidden_states = torch.cat([target_hidden_states_1, target_hidden_states_2], dim=0)
                        target_skeleton = self.cross_pose_attn(target_skeleton, source_hidden_states=target_hidden_states)
                        batch_frames, c_dim, h_dim, w_dim = target_skeleton.size()
                        target_skeleton = rearrange(target_skeleton, "(b f) c h w -> b c f h w", b=batch_frames//num_frames, f=num_frames, c=c_dim, h=h_dim, w=w_dim)
                        source_skeleton = torch.zeros_like(target_skeleton)
                        skeleton = torch.cat([torch.unsqueeze(source_skeleton[0], dim=0), torch.unsqueeze(target_skeleton[0], dim=0), torch.unsqueeze(source_skeleton[1], dim=0), torch.unsqueeze(target_skeleton[1], dim=0)], dim=0)
                    elif hidden_states.size()[0] == 2:
                        print(1/0)
                        skeleton = self.cross_pose_attn(target_skeleton, source_hidden_states=target_hidden_states)
                    else:
                        source_hidden_states = hidden_states
                        if self.detached_idx == 0:
                            batch, num_frames, c_dim, h_dim, w_dim = skeleton.size()
                            skeleton = rearrange(skeleton, "b f c h w -> (b f) c h w")
                            source_hidden_states = rearrange(source_hidden_states, "b c f h w -> (b f) c h w")
                        else:
                            batch, c_dim, num_frames, h_dim, w_dim = skeleton.size()
                            skeleton = rearrange(skeleton, "b c f h w -> (b f) c h w")
                            source_hidden_states = rearrange(source_hidden_states, "b c f h w -> (b f) c h w")
                        skeleton = self.cross_pose_attn(skeleton, source_hidden_states=source_hidden_states)
                        batch_frames, c_dim, h_dim, w_dim = skeleton.size()
                        skeleton = rearrange(skeleton, "(b f) c h w -> b c f h w", b=batch_frames//num_frames, f=num_frames, c=c_dim, h=h_dim, w=w_dim)
                    hidden_states = hidden_states + skeleton

                idx = idx + 1

            output_states += (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

            output_states += (hidden_states,)

        return hidden_states, output_states, skeleton
        # return hidden_states, output_states


class DownBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        output_scale_factor=1.0,
        add_downsample=True,
        downsample_padding=1,
        detached_idx=None,
    ):
        super().__init__()
        resnets = []

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )

        self.resnets = nn.ModuleList(resnets)

        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    Downsample2D(
                        out_channels, use_conv=True, out_channels=out_channels, padding=downsample_padding, name="op"
                    )
                ]
            )
        else:
            self.downsamplers = None

        self.gradient_checkpointing = False

        self.detached_idx = detached_idx
        # self.cross_pose_attn = Detached_Adapter(cin=int(3 * 64), channels=[320, 640, 1280, 1280][:4], nums_rb=2, ksize=1, sk=True, use_conv=False, idx=detached_idx)

    def forward(self, hidden_states, temb=None, encoder_hidden_states=None, skeleton=None, temb_aux=None, masks=None, source_masks=None, target_masks=None, rectangle_source_masks=None):
        output_states = ()
        idx = 1
        for resnet in self.resnets:
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(resnet), hidden_states, temb, temb_aux, masks)

                if (idx + 1) % 3 == 0 and skeleton is not None:
                    print("--------------------This is skeleton adapter in the DownBlock2D---------------------------")
                    print(self.detached_idx)
                    print(idx)
                    print(skeleton.size())
                    # print(encoder_hidden_states.size())
                    print(hidden_states.size())
                    print("----------------------------------------------------------------")
                    if hidden_states.size()[0] == 4:
                        if self.detached_idx == 0:
                            target_skeleton_1 = torch.unsqueeze(skeleton[1], dim=0)
                            num_frames = target_skeleton_1.size()[1]
                            target_skeleton_1 = rearrange(target_skeleton_1, "b f c h w -> (b f) c h w")
                            target_skeleton_2 = torch.unsqueeze(skeleton[3], dim=0)
                            target_skeleton_2 = rearrange(target_skeleton_2, "b f c h w -> (b f) c h w")
                            target_skeleton = torch.cat([target_skeleton_1, target_skeleton_2], dim=0)
                            target_hidden_states_1 = torch.unsqueeze(hidden_states[1], dim=0)
                            target_hidden_states_1 = rearrange(target_hidden_states_1, "b c f h w -> (b f) c h w")
                            target_hidden_states_2 = torch.unsqueeze(hidden_states[3], dim=0)
                            target_hidden_states_2 = rearrange(target_hidden_states_2, "b c f h w -> (b f) c h w")
                            target_hidden_states = torch.cat([target_hidden_states_1, target_hidden_states_2], dim=0)
                        else:
                            target_skeleton_1 = torch.unsqueeze(skeleton[1], dim=0)
                            num_frames = target_skeleton_1.size()[2]
                            target_skeleton_1 = rearrange(target_skeleton_1, "b c f h w -> (b f) c h w")
                            target_skeleton_2 = torch.unsqueeze(skeleton[3], dim=0)
                            target_skeleton_2 = rearrange(target_skeleton_2, "b c f h w -> (b f) c h w")
                            target_skeleton = torch.cat([target_skeleton_1, target_skeleton_2], dim=0)
                            target_hidden_states_1 = torch.unsqueeze(hidden_states[1], dim=0)
                            target_hidden_states_1 = rearrange(target_hidden_states_1, "b c f h w -> (b f) c h w")
                            target_hidden_states_2 = torch.unsqueeze(hidden_states[3], dim=0)
                            target_hidden_states_2 = rearrange(target_hidden_states_2, "b c f h w -> (b f) c h w")
                            target_hidden_states = torch.cat([target_hidden_states_1, target_hidden_states_2], dim=0)
                        target_skeleton = self.cross_pose_attn(target_skeleton, source_hidden_states=target_hidden_states)
                        batch_frames, c_dim, h_dim, w_dim = target_skeleton.size()
                        target_skeleton = rearrange(target_skeleton, "(b f) c h w -> b c f h w", b=batch_frames//num_frames, f=num_frames, c=c_dim, h=h_dim, w=w_dim)
                        source_skeleton = torch.zeros_like(target_skeleton)
                        skeleton = torch.cat([torch.unsqueeze(source_skeleton[0], dim=0), torch.unsqueeze(target_skeleton[0], dim=0), torch.unsqueeze(source_skeleton[1], dim=0), torch.unsqueeze(target_skeleton[1], dim=0)], dim=0)
                    elif hidden_states.size()[0] == 2:
                        print(1/0)
                        skeleton = self.cross_pose_attn(target_skeleton, source_hidden_states=target_hidden_states)
                    else:
                        source_hidden_states = hidden_states
                        if self.detached_idx == 0:
                            batch, num_frames, c_dim, h_dim, w_dim = skeleton.size()
                            skeleton = rearrange(skeleton, "b f c h w -> (b f) c h w")
                            source_hidden_states = rearrange(source_hidden_states, "b c f h w -> (b f) c h w")
                        else:
                            batch, c_dim, num_frames, h_dim, w_dim = skeleton.size()
                            skeleton = rearrange(skeleton, "b c f h w -> (b f) c h w")
                            source_hidden_states = rearrange(source_hidden_states, "b c f h w -> (b f) c h w")
                        skeleton = self.cross_pose_attn(skeleton, source_hidden_states=source_hidden_states)
                        batch_frames, c_dim, h_dim, w_dim = skeleton.size()
                        skeleton = rearrange(skeleton, "(b f) c h w -> b c f h w", b=batch_frames//num_frames, f=num_frames, c=c_dim, h=h_dim, w=w_dim)
                    hidden_states = hidden_states + skeleton

                idx = idx + 1

            else:
                hidden_states = resnet(hidden_states, temb, temb_aux, masks)

                if (idx + 1) % 3 == 0 and skeleton is not None:
                    print("--------------------This is skeleton adapter in the DownBlock2D---------------------------")
                    print(self.detached_idx)
                    print(idx)
                    print(skeleton.size())
                    # print(encoder_hidden_states.size())
                    print(hidden_states.size())
                    print("----------------------------------------------------------------")
                    if hidden_states.size()[0] == 4:
                        if self.detached_idx == 0:
                            target_skeleton_1 = torch.unsqueeze(skeleton[1], dim=0)
                            num_frames = target_skeleton_1.size()[1]
                            target_skeleton_1 = rearrange(target_skeleton_1, "b f c h w -> (b f) c h w")
                            target_skeleton_2 = torch.unsqueeze(skeleton[3], dim=0)
                            target_skeleton_2 = rearrange(target_skeleton_2, "b f c h w -> (b f) c h w")
                            target_skeleton = torch.cat([target_skeleton_1, target_skeleton_2], dim=0)
                            target_hidden_states_1 = torch.unsqueeze(hidden_states[1], dim=0)
                            target_hidden_states_1 = rearrange(target_hidden_states_1, "b c f h w -> (b f) c h w")
                            target_hidden_states_2 = torch.unsqueeze(hidden_states[3], dim=0)
                            target_hidden_states_2 = rearrange(target_hidden_states_2, "b c f h w -> (b f) c h w")
                            target_hidden_states = torch.cat([target_hidden_states_1, target_hidden_states_2], dim=0)
                        else:
                            target_skeleton_1 = torch.unsqueeze(skeleton[1], dim=0)
                            num_frames = target_skeleton_1.size()[2]
                            target_skeleton_1 = rearrange(target_skeleton_1, "b c f h w -> (b f) c h w")
                            target_skeleton_2 = torch.unsqueeze(skeleton[3], dim=0)
                            target_skeleton_2 = rearrange(target_skeleton_2, "b c f h w -> (b f) c h w")
                            target_skeleton = torch.cat([target_skeleton_1, target_skeleton_2], dim=0)
                            target_hidden_states_1 = torch.unsqueeze(hidden_states[1], dim=0)
                            target_hidden_states_1 = rearrange(target_hidden_states_1, "b c f h w -> (b f) c h w")
                            target_hidden_states_2 = torch.unsqueeze(hidden_states[3], dim=0)
                            target_hidden_states_2 = rearrange(target_hidden_states_2, "b c f h w -> (b f) c h w")
                            target_hidden_states = torch.cat([target_hidden_states_1, target_hidden_states_2], dim=0)
                        target_skeleton = self.cross_pose_attn(target_skeleton, source_hidden_states=target_hidden_states)
                        batch_frames, c_dim, h_dim, w_dim = target_skeleton.size()
                        target_skeleton = rearrange(target_skeleton, "(b f) c h w -> b c f h w", b=batch_frames//num_frames, f=num_frames, c=c_dim, h=h_dim, w=w_dim)
                        source_skeleton = torch.zeros_like(target_skeleton)
                        skeleton = torch.cat([torch.unsqueeze(source_skeleton[0], dim=0), torch.unsqueeze(target_skeleton[0], dim=0), torch.unsqueeze(source_skeleton[1], dim=0), torch.unsqueeze(target_skeleton[1], dim=0)], dim=0)
                    elif hidden_states.size()[0] == 2:
                        print(1/0)
                        skeleton = self.cross_pose_attn(target_skeleton, source_hidden_states=target_hidden_states)
                    else:
                        source_hidden_states = hidden_states
                        if self.detached_idx == 0:
                            batch, num_frames, c_dim, h_dim, w_dim = skeleton.size()
                            skeleton = rearrange(skeleton, "b f c h w -> (b f) c h w")
                            source_hidden_states = rearrange(source_hidden_states, "b c f h w -> (b f) c h w")
                        else:
                            batch, c_dim, num_frames, h_dim, w_dim = skeleton.size()
                            skeleton = rearrange(skeleton, "b c f h w -> (b f) c h w")
                            source_hidden_states = rearrange(source_hidden_states, "b c f h w -> (b f) c h w")
                        skeleton = self.cross_pose_attn(skeleton, source_hidden_states=source_hidden_states)
                        batch_frames, c_dim, h_dim, w_dim = skeleton.size()
                        skeleton = rearrange(skeleton, "(b f) c h w -> b c f h w", b=batch_frames//num_frames, f=num_frames, c=c_dim, h=h_dim, w=w_dim)
                    hidden_states = hidden_states + skeleton

                idx = idx + 1

            output_states += (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

            output_states += (hidden_states,)

        return hidden_states, output_states, skeleton
        # return hidden_states, output_states


class CrossAttnUpBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        prev_output_channel: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        attn_num_head_channels=1,
        cross_attention_dim=1280,
        output_scale_factor=1.0,
        add_upsample=True,
        dual_cross_attention=False,
        use_linear_projection=False,
        only_cross_attention=False,
        upcast_attention=False,
        use_sc_attn=False,
        use_st_attn=False,
    ):
        super().__init__()
        resnets = []
        attentions = []

        self.has_cross_attention = True
        self.attn_num_head_channels = attn_num_head_channels

        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            resnets.append(
                ResnetBlock2D(
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )
            if dual_cross_attention:
                raise NotImplementedError
            attentions.append(
                Transformer2DModel(
                    attn_num_head_channels,
                    out_channels // attn_num_head_channels,
                    in_channels=out_channels,
                    num_layers=1,
                    cross_attention_dim=cross_attention_dim,
                    norm_num_groups=resnet_groups,
                    use_linear_projection=use_linear_projection,
                    only_cross_attention=only_cross_attention,
                    upcast_attention=upcast_attention,
                    use_sc_attn=use_sc_attn,
                    use_st_attn=True if (use_st_attn and i == 0) else False,
                )
            )

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

        if add_upsample:
            self.upsamplers = nn.ModuleList([Upsample2D(out_channels, use_conv=True, out_channels=out_channels)])
        else:
            self.upsamplers = None

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states,
        res_hidden_states_tuple,
        temb=None,
        encoder_hidden_states=None,
        upsample_size=None,
        attention_mask=None,
        normal_infer=False,
        temb_aux=None,
        masks=None,
        source_masks=None,
        target_masks=None,
        rectangle_source_masks=None,
    ):
        for resnet, attn in zip(self.resnets, self.attentions):
            # pop res hidden states
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None, normal_infer=False):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict, normal_infer=normal_infer)
                        else:
                            return module(*inputs)

                    return custom_forward

                hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(resnet), hidden_states, temb, temb_aux, masks, source_masks, target_masks)
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(attn, return_dict=False, normal_infer=normal_infer),
                    hidden_states,
                    encoder_hidden_states,
                    source_masks,
                    target_masks,
                    rectangle_source_masks,
                )[0]
            else:
                hidden_states = resnet(hidden_states, temb, temb_aux, masks, source_masks=source_masks, target_masks=target_masks)
                hidden_states = attn(hidden_states, encoder_hidden_states=encoder_hidden_states, normal_infer=normal_infer, source_masks=source_masks, target_masks=target_masks, rectangle_source_masks=rectangle_source_masks).sample

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, upsample_size)

        return hidden_states


class UpBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        prev_output_channel: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        output_scale_factor=1.0,
        add_upsample=True,
    ):
        super().__init__()
        resnets = []

        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            resnets.append(
                ResnetBlock2D(
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )

        self.resnets = nn.ModuleList(resnets)

        if add_upsample:
            self.upsamplers = nn.ModuleList([Upsample2D(out_channels, use_conv=True, out_channels=out_channels)])
        else:
            self.upsamplers = None

        self.gradient_checkpointing = False

    def forward(self, hidden_states, res_hidden_states_tuple, temb=None, upsample_size=None, temb_aux=None, masks=None, source_masks=None, target_masks=None, rectangle_source_masks=None):
        for resnet in self.resnets:
            # pop res hidden states
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(resnet), hidden_states, temb, temb_aux, masks, source_masks, target_masks)
            else:
                hidden_states = resnet(hidden_states, temb, temb_aux, masks, source_masks=source_masks, target_masks=target_masks)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, upsample_size)

        return hidden_states
