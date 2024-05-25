import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, Tuple, List, Callable, Dict
from torchvision.utils import save_image
from einops import rearrange, repeat
from diffusers.utils.import_utils import is_xformers_available


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

class TemporalAttentionBase:
    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0

    def after_step(self):
        pass

    def __call__(self, q, k, v, sim=None, attn=None, is_cross=None, place_in_unet=None, num_heads=None, attention_mask=None, **kwargs):
        out = self.forward(q=q, k=k, v=v, is_cross=is_cross, place_in_unet=place_in_unet, num_heads=num_heads, attention_mask=attention_mask, sim=sim, attn=attn, **kwargs)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            # after step
            self.after_step()
        return out

    def forward(self, q, k, v, sim=None, attn=None, is_cross=None, place_in_unet=None, num_heads=None, attention_mask=None, **kwargs):
        print("We are in the TemporalAttentionBase !!!!!!")
        if not is_cross:
            print("We are in the temporal self attention")

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
        else:
            print("We are in the temporal cross attention")
            print(1/0)
            attn = attn.to(v.dtype)
            out = torch.bmm(attn, v)
            out = reshape_batch_dim_to_heads_base(out, num_heads)
        return out

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0


def regiter_temporal_attention_editor_diffusers(model, editor: TemporalAttentionBase):
    """
    Register a attention editor to Diffuser Pipeline, refer from [Prompt-to-Prompt]
    """
    def ca_forward(self, place_in_unet):
        def forward(hidden_states, encoder_hidden_states=None, attention_mask=None):

            batch_size, sequence_length, _ = hidden_states.shape
            is_cross = encoder_hidden_states is not None
            encoder_hidden_states = encoder_hidden_states
            if self.group_norm is not None:
                hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

            if not is_cross:
                query = self.to_q(hidden_states)
                dim = query.shape[-1]
                query = self.reshape_heads_to_batch_dim(query)
                if self.added_kv_proj_dim is not None:
                    raise NotImplementedError
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

                assert self._slice_size is None or query.shape[0] // self._slice_size == 1
                if self.upcast_attention:
                    query = query.float()
                    key = key.float()

                if attention_mask is None:
                    print(1/0)
                out = editor(q=query, k=key, v=value, is_cross=is_cross, place_in_unet=place_in_unet, num_heads=self.heads, attention_mask=attention_mask, scale=self.scale)
                hidden_states = self.to_out[0](out)
                hidden_states = self.to_out[1](hidden_states)
                return hidden_states
            else:
                print(1/0)

        return forward

    def register_editor(net_, count, place_in_unet):
        if net_.__class__.__name__ == 'TemporalSelfAttention':
            net_.forward = ca_forward(net_, place_in_unet)
            return count + 1
        elif hasattr(net_, 'children'):
            for net__ in net_.children():
                count = register_editor(net__, count, place_in_unet)
        return count


    cross_att_count = 0
    for net_name, net in model.unet.named_children():
        if "down" in net_name:
            cross_att_count += register_editor(net, 0, "down")
        elif "mid" in net_name:
            cross_att_count += register_editor(net, 0, "mid")
        elif "up" in net_name:
            cross_att_count += register_editor(net, 0, "up")
    editor.num_att_layers = cross_att_count
    print(editor.num_att_layers)
