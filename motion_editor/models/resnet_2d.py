# Adapted from https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/resnet.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange


class TemporalConv(nn.Conv1d):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = True, padding_mode: str = 'zeros', device=None, dtype=None) -> None:
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, device, dtype)

        # nn.init.dirac_(self.weight.data) # initialized to be identity
        nn.init.zeros_(self.weight.data) # initialized to zeros
        nn.init.zeros_(self.bias.data)

    def forward(self, x):
        # ipdb.set_trace()
        _, c_dim, f_dim, h_dim, w_dim = x.size()

        x = rearrange(x, 'b c f h w -> (b h w) c f')
        x = super().forward(x)
        x = rearrange(x, "(b h w) c f -> b c f h w", h=h_dim, w=w_dim)

        return x

class InflatedConv3d(nn.Conv2d):
    def forward(self, x):
        video_length = x.shape[2]

        x = rearrange(x, "b c f h w -> (b f) c h w")
        x = super().forward(x)
        x = rearrange(x, "(b f) c h w -> b c f h w", f=video_length)

        return x


class Upsample2D(nn.Module):
    def __init__(self, channels, use_conv=False, use_conv_transpose=False, out_channels=None, name="conv"):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_conv_transpose = use_conv_transpose
        self.name = name

        conv = None
        if use_conv_transpose:
            raise NotImplementedError
        elif use_conv:
            conv = InflatedConv3d(self.channels, self.out_channels, 3, padding=1)

        if name == "conv":
            self.conv = conv
        else:
            self.Conv2d_0 = conv

    def forward(self, hidden_states, output_size=None):
        assert hidden_states.shape[1] == self.channels

        if self.use_conv_transpose:
            raise NotImplementedError

        # Cast to float32 to as 'upsample_nearest2d_out_frame' op does not support bfloat16
        dtype = hidden_states.dtype
        if dtype == torch.bfloat16:
            hidden_states = hidden_states.to(torch.float32)

        # upsample_nearest_nhwc fails with large batch sizes. see https://github.com/huggingface/diffusers/issues/984
        if hidden_states.shape[0] >= 64:
            hidden_states = hidden_states.contiguous()

        # if `output_size` is passed we force the interpolation output
        # size and do not make use of `scale_factor=2`
        if output_size is None:
            hidden_states = F.interpolate(hidden_states, scale_factor=[1.0, 2.0, 2.0], mode="nearest")
        else:
            hidden_states = F.interpolate(hidden_states, size=output_size, mode="nearest")

        # If the input is bfloat16, we cast back to bfloat16
        if dtype == torch.bfloat16:
            hidden_states = hidden_states.to(dtype)

        if self.use_conv:
            if self.name == "conv":
                hidden_states = self.conv(hidden_states)
            else:
                hidden_states = self.Conv2d_0(hidden_states)

        return hidden_states


class Downsample2D(nn.Module):
    def __init__(self, channels, use_conv=False, out_channels=None, padding=1, name="conv"):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.padding = padding
        stride = 2
        self.name = name

        if use_conv:
            conv = InflatedConv3d(self.channels, self.out_channels, 3, stride=stride, padding=padding)
        else:
            raise NotImplementedError

        if name == "conv":
            self.Conv2d_0 = conv
            self.conv = conv
        elif name == "Conv2d_0":
            self.conv = conv
        else:
            self.conv = conv

    def forward(self, hidden_states):
        assert hidden_states.shape[1] == self.channels
        if self.use_conv and self.padding == 0:
            raise NotImplementedError

        assert hidden_states.shape[1] == self.channels
        hidden_states = self.conv(hidden_states)

        return hidden_states


class ResnetBlock2D(nn.Module):
    def __init__(
        self,
        *,
        in_channels,
        out_channels=None,
        conv_shortcut=False,
        dropout=0.0,
        temb_channels=512,
        groups=32,
        groups_out=None,
        pre_norm=True,
        eps=1e-6,
        non_linearity="swish",
        time_embedding_norm="default",
        output_scale_factor=1.0,
        use_in_shortcut=None,
        temporal_conv=True,
    ):
        super().__init__()
        self.pre_norm = pre_norm
        self.pre_norm = True
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.time_embedding_norm = time_embedding_norm
        self.output_scale_factor = output_scale_factor

        if groups_out is None:
            groups_out = groups

        self.norm1 = torch.nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=eps, affine=True)

        self.conv1 = InflatedConv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if temb_channels is not None:
            if self.time_embedding_norm == "default":
                time_emb_proj_out_channels = out_channels
            elif self.time_embedding_norm == "scale_shift":
                time_emb_proj_out_channels = out_channels * 2
            else:
                raise ValueError(f"unknown time_embedding_norm : {self.time_embedding_norm} ")

            self.time_emb_proj = torch.nn.Linear(temb_channels, time_emb_proj_out_channels)
        else:
            self.time_emb_proj = None

        self.norm2 = torch.nn.GroupNorm(num_groups=groups_out, num_channels=out_channels, eps=eps, affine=True)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = InflatedConv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if non_linearity == "swish":
            self.nonlinearity = lambda x: F.silu(x)
        elif non_linearity == "mish":
            self.nonlinearity = Mish()
        elif non_linearity == "silu":
            self.nonlinearity = nn.SiLU()

        self.use_in_shortcut = self.in_channels != self.out_channels if use_in_shortcut is None else use_in_shortcut

        self.conv_shortcut = None
        if self.use_in_shortcut:
            self.conv_shortcut = InflatedConv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

        self.temp_conv1, self.temp_conv2 = None, None

        if temporal_conv:
            self.temp_conv1 = TemporalConv(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
            self.temp_conv2 = TemporalConv(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, input_tensor, temb, temb_aux, masks, source_masks=None, target_masks=None, rectangle_source_masks=None):
        hidden_states = input_tensor

        hidden_states = self.norm1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)

        hidden_states = self.conv1(hidden_states)

        if self.temp_conv1 is not None:
            hidden_states = hidden_states + self.temp_conv1(hidden_states)

        if temb is not None:
            temb = self.time_emb_proj(self.nonlinearity(temb))[:, :, None, None, None]

            video_length = hidden_states.size(2)
            if temb.size(0) == hidden_states.size(0) * video_length:
                temb = temb.reshape(hidden_states.size(0), video_length, temb.size(1), 1, 1)  # b,f,c,1,1
                temb = temb.permute(0, 2, 1, 3, 4)  # b,c,f,1,1
            ## NOTE the shape of temb, hidden state and masks here # it is b,c,f,h,w
            if masks is not None:
                masks = torch.nn.functional.interpolate(masks, size=hidden_states.size()[-3:], mode="nearest")
            if temb_aux is not None:  ## this keeps the same no matter how many masks
                temb_aux = self.time_emb_proj(self.nonlinearity(temb_aux))[:, :, None, None, None]  # B,C,1,1,1
                if temb_aux.size(0) == hidden_states.size(0) * video_length:
                    temb_aux = temb_aux.reshape(hidden_states.size(0), video_length, temb_aux.size(1), 1, 1)  # b,f,c,1,1
                    temb_aux = temb_aux.permute(0, 2, 1, 3, 4)
                temb = temb * masks + (1 - masks) * temb_aux

        if temb is not None and self.time_embedding_norm == "default":
            hidden_states = hidden_states + temb

        hidden_states = self.norm2(hidden_states)

        if temb is not None and self.time_embedding_norm == "scale_shift":
            scale, shift = torch.chunk(temb, 2, dim=1)
            hidden_states = hidden_states * (1 + scale) + shift

        hidden_states = self.nonlinearity(hidden_states)

        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.temp_conv2 is not None:
            hidden_states = hidden_states + self.temp_conv2(hidden_states)

        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor)

        output_tensor = (input_tensor + hidden_states) / self.output_scale_factor

        return output_tensor


class Mish(torch.nn.Module):
    def forward(self, hidden_states):
        return hidden_states * torch.tanh(torch.nn.functional.softplus(hidden_states))




