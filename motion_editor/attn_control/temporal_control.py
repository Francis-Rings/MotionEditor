from diffusers.utils.import_utils import is_xformers_available
import os
import torch
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from torchvision.utils import save_image

from motion_editor.attn_control.temporal_control_utils import TemporalAttentionBase

if is_xformers_available():
    import xformers
    import xformers.ops
else:
    xformers = None

def reshape_batch_dim_to_heads_base(tensor, heads):
    batch_size, seq_len, dim = tensor.shape
    head_size = heads
    tensor = tensor.reshape(batch_size // head_size, head_size, seq_len, dim)
    tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size // head_size, seq_len, dim * head_size)
    return tensor


class TemporalSelfAttentionControl(TemporalAttentionBase):
    MODEL_TYPE = {
        "SD": 16,
        "SDXL": 70
    }

    def __init__(self, start_step=4, start_layer=10, layer_idx=None, step_idx=None, total_steps=50, model_type="SD"):
        super().__init__()
        self.total_steps = total_steps
        self.total_layers = self.MODEL_TYPE.get(model_type, 16)
        self.start_step = start_step
        self.start_layer = start_layer
        self.layer_idx = layer_idx if layer_idx is not None else list(range(start_layer, self.total_layers))
        self.step_idx = step_idx if step_idx is not None else list(range(start_step, total_steps))
        print("TemporalSelfAttentionControl at denoising steps: ", self.step_idx)
        print("TemporalSelfAttentionControl at U-Net layers: ", self.layer_idx)

    def attn_batch(self, q, k, v, num_heads, sim=None, attn=None, is_cross=None, place_in_unet=None, attention_mask=None, **kwargs):

        attention_scores = torch.baddbmm(
            torch.empty(q.shape[0], q.shape[1], k.shape[1], dtype=q.dtype, device=q.device),
            q,
            k.transpose(-1, -2),
            beta=0,
            alpha=kwargs.get("scale"),
        )
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        attention_probs = attention_scores.softmax(dim=-1)
        attention_probs = attention_probs.to(v.dtype)
        hidden_states = torch.bmm(attention_probs, v)
        out = reshape_batch_dim_to_heads_base(hidden_states, num_heads)

        return out

    def xformers_based_attn_batch(self, q, k, v, num_heads, sim=None, attn=None, is_cross=None, place_in_unet=None, attention_mask=None,**kwargs):

        query = q.contiguous()
        key = k.contiguous()
        value = v.contiguous()
        out = xformers.ops.memory_efficient_attention(query, key, value, attn_bias=attention_mask)
        out = out.to(query.dtype)
        out = reshape_batch_dim_to_heads_base(out, num_heads)
        return out

    def forward(self, q, k, v, sim=None, attn=None, is_cross=None, place_in_unet=None, num_heads=None, attention_mask=None, **kwargs):
        """
        Attention forward function
        """
        if is_cross or self.cur_step not in self.step_idx or self.cur_att_layer not in self.layer_idx:
            return super().forward(q=q, k=k, v=v, is_cross=is_cross, place_in_unet=place_in_unet, num_heads=num_heads, sim=sim, attn=attn, attention_mask=attention_mask, **kwargs)

        qu, qc = q.chunk(2)
        ku, kc = k.chunk(2)
        vu, vc = v.chunk(2)
        num_frames = q.size()[0] // num_heads // 4

        out_u_s = self.attn_batch(q=qu[:num_heads * num_frames], k=ku[:num_heads * num_frames], v=vu[:num_heads * num_frames], num_heads=num_heads, attention_mask=attention_mask, **kwargs)
        out_u_t = self.attn_batch(q=qu[-num_heads * num_frames:], k=ku[:num_heads * num_frames], v=vu[:num_heads * num_frames], num_heads=num_heads, attention_mask=attention_mask, **kwargs)
        out_c_s = self.attn_batch(q=qc[:num_heads * num_frames], k=kc[:num_heads * num_frames], v=vc[:num_heads * num_frames], num_heads=num_heads, attention_mask=attention_mask, **kwargs)
        out_c_t = self.attn_batch(q=qc[-num_heads * num_frames:], k=kc[:num_heads * num_frames], v=vc[:num_heads * num_frames], num_heads=num_heads, attention_mask=attention_mask, **kwargs)
        out = torch.cat([out_u_s, out_u_t, out_c_s, out_c_t], dim=0)

        return out

