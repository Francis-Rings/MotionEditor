from typing import Optional

import torch
from torch import nn
from einops import rearrange, repeat
import torch.nn.functional as F
from diffusers.models.attention import FeedForward
from diffusers.utils.import_utils import is_xformers_available

from motion_editor.models.adapter_self_temporal_attn import AdapterSelfTemporalCrossAttention

if is_xformers_available():
    import xformers
    import xformers.ops
else:
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

    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None):
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


class CrossPoseAttn(nn.Module):
    def __init__(self, dim, num_attention_heads, cross_source_attention_dim):
        super().__init__()
        self.query_dim = dim
        self.num_attention_heads = num_attention_heads
        self.cross_source_attention_dim = cross_source_attention_dim
        self.cross_pose_attn = AdapterCrossAttention(
            query_dim=self.query_dim,
            cross_attention_dim=self.cross_source_attention_dim,
            heads=num_attention_heads,
            dim_head=self.query_dim//num_attention_heads,
            dropout=0.0,
            bias=False,
            upcast_attention=False)
        self.cross_pose_norm = nn.LayerNorm(self.query_dim)

    def forward(self, x, source_encoder_hidden_states):
        batch, c_dim, num_frames, h_dim, w_dim = x.size()
        if x.size()[0] != source_encoder_hidden_states.size()[0]:
            source_encoder_hidden_states = source_encoder_hidden_states[-1]
            source_encoder_hidden_states = torch.unsqueeze(source_encoder_hidden_states, dim=0)
            # x = torch.cat([x]*2)
            # batch = batch * 2
        x = rearrange(x, "b c f h w -> (b f) (h w) c", b=batch, c=c_dim, f=num_frames, h=h_dim, w=w_dim)
        source_encoder_hidden_states = rearrange(source_encoder_hidden_states, "b c f h w -> (b f) (h w) c", b=batch, c=c_dim, f=num_frames, h=h_dim, w=w_dim)
        out = self.cross_pose_norm(x)
        out = self.cross_pose_attn(out, encoder_hidden_states=source_encoder_hidden_states) + out
        out = rearrange(out, "(b f) (h w) c -> b c f h w", b=batch, c=c_dim, f=num_frames, h=h_dim, w=w_dim)
        return out


class AdapterSpatialTemporalAttention(AdapterCrossAttention):
    def forward_dense_attn(self, hidden_states, encoder_hidden_states=None, attention_mask=None, video_length=None):
        batch_size, sequence_length, _ = hidden_states.shape

        encoder_hidden_states = encoder_hidden_states

        if self.group_norm is not None:
            hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = self.to_q(hidden_states)
        dim = query.shape[-1]
        query = self.reshape_heads_to_batch_dim(query)

        if self.added_kv_proj_dim is not None:
            raise NotImplementedError

        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        key = self.to_k(encoder_hidden_states)
        value = self.to_v(encoder_hidden_states)

        key = rearrange(key, "(b f) n d -> b f n d", f=video_length)
        key = key.unsqueeze(1).repeat(1, video_length, 1, 1, 1)  # (b f f n d)
        key = rearrange(key, "b f g n d -> (b f) (g n) d")

        value = rearrange(value, "(b f) n d -> b f n d", f=video_length)
        value = value.unsqueeze(1).repeat(1, video_length, 1, 1, 1)  # (b f f n d)
        value = rearrange(value, "b f g n d -> (b f) (g n) d")

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

    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, video_length=None,
                normal_infer=False):
        if normal_infer:
            return super().forward(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                # video_length=video_length,
            )
        else:
            return self.forward_dense_attn(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                video_length=video_length,
            )


class AdapterSparseCausalAttention(AdapterCrossAttention):
    def forward_sc_attn(self, hidden_states, encoder_hidden_states=None, attention_mask=None, video_length=None, source_masks=None, target_masks=None, rectangle_source_masks=None):
        batch_size, sequence_length, _ = hidden_states.shape

        encoder_hidden_states = encoder_hidden_states

        if self.group_norm is not None:
            hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = self.to_q(hidden_states)
        dim = query.shape[-1]
        query = self.reshape_heads_to_batch_dim(query)

        if self.added_kv_proj_dim is not None:
            raise NotImplementedError

        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        key = self.to_k(encoder_hidden_states)
        value = self.to_v(encoder_hidden_states)

        former_frame_index = torch.arange(video_length) - 1
        former_frame_index[0] = 0

        key = rearrange(key, "(b f) d c -> b f d c", f=video_length)
        key = torch.cat([key[:, [0] * video_length], key[:, former_frame_index]], dim=2)
        key = rearrange(key, "b f d c -> (b f) d c")

        value = rearrange(value, "(b f) d c -> b f d c", f=video_length)
        value = torch.cat([value[:, [0] * video_length], value[:, former_frame_index]], dim=2)
        value = rearrange(value, "b f d c -> (b f) d c")

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

    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, video_length=None, normal_infer=False, source_masks=None, target_masks=None, rectangle_source_masks=None):
        if normal_infer:
            return super().forward(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                # video_length=video_length,
            )
        else:
            return self.forward_sc_attn(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                video_length=video_length,
                source_masks=source_masks,
                target_masks=target_masks,
                rectangle_source_masks=rectangle_source_masks,
            )



class TemporalConv(nn.Conv1d):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0,
                 dilation: int = 1, groups: int = 1, bias: bool = True, padding_mode: str = 'zeros', device=None,
                 dtype=None, num_frames=8) -> None:
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode,
                         device, dtype)
        # nn.init.dirac_(self.weight.data) # initialized to be identity
        nn.init.zeros_(self.weight.data)  # initialized to zeros
        nn.init.zeros_(self.bias.data)
        self.num_frames = num_frames

    def forward(self, x):
        # ipdb.set_trace()
        batch_frames, c_dim, h_dim, w_dim = x.size()
        x = rearrange(x, "(b f) c h w -> b c f h w", b=batch_frames // self.num_frames, f=self.num_frames, c=c_dim,
                      h=h_dim, w=w_dim)
        _, c_dim, f_dim, h_dim, w_dim = x.size()

        x = rearrange(x, 'b c f h w -> (b h w) c f')
        x = super().forward(x)
        # x = rearrange(x, "(b h w) c f -> b c f h w", h=h_dim, w=w_dim)
        x = rearrange(x, "(b h w) c f -> (b f) c h w", h=h_dim, w=w_dim, f=self.num_frames)

        return x


class ResnetBlock(nn.Module):
    def __init__(self, in_c, out_c, down, ksize=3, sk=False, use_conv=True, num_attention_heads=8, num_frames=8, idx=None):
        super().__init__()
        self.idx = idx
        ps = ksize // 2

        if self.idx <= 3:
            out_c = 320
        elif 3 < self.idx <= 6:
            out_c = 640
        else:
            out_c = 1280

        # self.block1 = nn.Conv2d(out_c, out_c, 3, 1, 1)
        self.block1 = TemporalConv(out_c, out_c, kernel_size=3, stride=1, padding=1)
        self.act = nn.ReLU()
        # self.block2 = nn.Conv2d(out_c, out_c, ksize, 1, ps)
        self.block2 = TemporalConv(out_c, out_c, kernel_size=ksize, stride=1, padding=ps)
        if sk == False:
            self.skep = nn.Conv2d(in_c, out_c, ksize, 1, ps)
        else:
            self.skep = None

        self.attn_temp = AdapterSparseCausalAttention(
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
        self.num_frames = num_frames

        # Cross Attention between source video feature and pose feature
        self.attn_pose = AdapterCrossAttention(
            query_dim=out_c,
            cross_attention_dim=out_c,
            heads=num_attention_heads,
            dim_head=out_c // num_attention_heads,
            dropout=0.0,
            bias=False,
            upcast_attention=False)
        self.cross_pose_norm = nn.LayerNorm(out_c)

        self.attn_self_temp = AdapterSelfTemporalCrossAttention(
            query_dim=out_c,
            heads=num_attention_heads,
            dim_head=out_c // num_attention_heads,
            dropout=0.0,
            bias=False,
            upcast_attention=False,
        )
        nn.init.zeros_(self.attn_self_temp.to_out[0].weight.data)
        self.norm_self_temp = nn.LayerNorm(out_c)


    def forward(self, x, source_hidden_states=None, source_masks=None, target_masks=None, rectangle_source_masks=None):

        b, c, t, h, w = x.shape
        x = rearrange(x, "b c t h w -> (b t) c h w", b=b, t=t)
        x_attn_branch = x

        h = self.block1(x)
        h = self.act(h)
        h = self.block2(h)

        if self.skep is not None:
            h = h + self.skep(x)
        else:
            h = h + x

        batch_frames, c_dim, h_dim, w_dim = x_attn_branch.size()
        x_attn_branch = rearrange(x_attn_branch, "(b f) c h w -> (b f) (h w) c", f=self.num_frames, b=batch_frames // self.num_frames, c=c_dim, h=h_dim, w=w_dim)
        norm_hidden_states = (self.norm_temp(x_attn_branch))
        attn_hidden_states = self.attn_temp(norm_hidden_states, video_length=self.num_frames, normal_infer=False) + x_attn_branch
        source_batch, source_c_dim, source_num_frames, source_h_dim, source_w_dim = source_hidden_states.size()
        source_hidden_states = rearrange(source_hidden_states, "b c f h w -> (b f) (h w) c", b=source_batch, c=source_c_dim, f=source_num_frames, h=source_h_dim, w=source_w_dim)
        attn_hidden_states = self.cross_pose_norm(attn_hidden_states)
        attn_hidden_states = self.attn_pose(attn_hidden_states, encoder_hidden_states=source_hidden_states) + attn_hidden_states
        attn_hidden_states = self.ff(self.ff_norm(attn_hidden_states)) + attn_hidden_states

        d = attn_hidden_states.shape[1]
        attn_hidden_states = rearrange(attn_hidden_states, "(b f) d c -> (b d) f c", f=t)
        attn_norm_hidden_states = (self.norm_self_temp(attn_hidden_states))
        causal_mask = torch.tril(torch.ones((attn_hidden_states.size(1), attn_hidden_states.size(1)), dtype=attn_hidden_states.dtype, device=attn_hidden_states.device))  # f,f
        causal_mask = (1.0 - causal_mask[None]) * -10000.0  # 1,f,f
        attn_hidden_states = self.attn_self_temp(attn_norm_hidden_states, attention_mask=causal_mask) + attn_hidden_states
        attn_hidden_states = rearrange(attn_hidden_states, "(b d) f c -> (b f) d c", d=d)

        attn_hidden_states = rearrange(attn_hidden_states, "(b f) (h w) c -> (b f) c h w", h=h_dim, w=w_dim, f=self.num_frames)

        hidden_states = attn_hidden_states + h

        return hidden_states


class ControlAdapter(nn.Module):
    def __init__(self, channels=[320, 640, 1280, 1280], nums_rb=3, cin=64, ksize=3, sk=False, use_conv=True, idx=None):
        super().__init__()
        self.channels = channels
        self.nums_rb = nums_rb
        self.body = []
        for i in range(len(channels)):
            for j in range(nums_rb):
                idx = i * self.nums_rb + j
                if (i != 0) and (j == 0):
                    self.body.append(
                        ResnetBlock(channels[i - 1], channels[i], down=True, ksize=ksize, sk=sk, use_conv=use_conv, idx=idx))
                else:
                    self.body.append(
                        ResnetBlock(channels[i], channels[i], down=False, ksize=ksize, sk=sk, use_conv=use_conv, idx=idx))
        self.body = nn.ModuleList(self.body)

    def forward(self, x_list, source_hidden_states=None, source_masks=None, target_masks=None, rectangle_source_masks=None):
        x = x_list[0]
        b, c, t, h, w = x.shape
        features = []
        for i in range(len(self.channels)):
            for j in range(self.nums_rb):
                idx = i * self.nums_rb + j
                out = self.body[idx](x_list[idx], source_hidden_states=source_hidden_states[idx], source_masks=None, target_masks=None, rectangle_source_masks=None)
                features.append(out)

        features = [rearrange(fn, '(b t) c h w -> b c t h w', b=b, t=t) for fn in features]
        return features

