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

class MutualAttentionBase:
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
        print("We are in the AttentionBase !!!!!!")
        # out = torch.einsum('b i j, b j d -> b i d', attn, v)
        # out = rearrange(out, '(b h) n d -> b n (h d)', h=num_heads)
        if not is_cross:
            print("We are in the self attention")
            query = q.contiguous()
            key = k.contiguous()
            value = v.contiguous()
            out = xformers.ops.memory_efficient_attention(query, key, value, attn_bias=attention_mask)
            out = out.to(query.dtype)
            out = reshape_batch_dim_to_heads_base(out, num_heads)
        else:
            print("We are in the cross attention")
            print(attn.size())
            attn = attn.to(v.dtype)
            out = torch.bmm(attn, v)
            out = reshape_batch_dim_to_heads_base(out, num_heads)
        return out

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0



class MutualAttentionStore(MutualAttentionBase):
    def __init__(self, res=[32], min_step=0, max_step=1000):
        super().__init__()
        self.res = res
        self.min_step = min_step
        self.max_step = max_step
        self.valid_steps = 0

        self.self_attns = []  # store the all attns
        self.cross_attns = []

        self.self_attns_step = []  # store the attns in each step
        self.cross_attns_step = []

    def after_step(self):
        if self.cur_step > self.min_step and self.cur_step < self.max_step:
            self.valid_steps += 1
            if len(self.self_attns) == 0:
                self.self_attns = self.self_attns_step
                self.cross_attns = self.cross_attns_step
            else:
                for i in range(len(self.self_attns)):
                    self.self_attns[i] += self.self_attns_step[i]
                    self.cross_attns[i] += self.cross_attns_step[i]
        self.self_attns_step.clear()
        self.cross_attns_step.clear()

    def forward(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        if attn.shape[1] <= 64 ** 2:  # avoid OOM
            if is_cross:
                self.cross_attns_step.append(attn)
            else:
                self.self_attns_step.append(attn)
        return super().forward(q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs)


def regiter_mutual_attention_editor_diffusers(model, editor: MutualAttentionBase):
    """
    Register a attention editor to Diffuser Pipeline, refer from [Prompt-to-Prompt]
    """
    def ca_forward(self, place_in_unet):
        def forward(hidden_states, encoder_hidden_states=None, attention_mask=None, video_length=None, normal_infer=False, source_masks=None, target_masks=None, rectangle_source_masks=None):

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

                # cur_frame_index = torch.arange(video_length)
                # key = rearrange(key, "(b f) d c -> b f d c", f=video_length)
                # key = key[:, cur_frame_index]
                # key = rearrange(key, "b f d c -> (b f) d c")
                # value = rearrange(value, "(b f) d c -> b f d c", f=video_length)
                # value = value[:, cur_frame_index]
                # value = rearrange(value, "b f d c -> (b f) d c")
                # key = self.reshape_heads_to_batch_dim(key)
                # value = self.reshape_heads_to_batch_dim(value)

                if attention_mask is not None:
                    if attention_mask.shape[-1] != query.shape[1]:
                        target_length = query.shape[1]
                        attention_mask = F.pad(attention_mask, (0, target_length), value=0.0)
                        attention_mask = attention_mask.repeat_interleave(self.heads, dim=0)

                assert self._slice_size is None or query.shape[0] // self._slice_size == 1
                if self.upcast_attention:
                    query = query.float()
                    key = key.float()

                out = editor(q=query, k=key, v=value, is_cross=is_cross, place_in_unet=place_in_unet, num_heads=self.heads, attention_mask=attention_mask, scale=self.scale)
                hidden_states = self.to_out[0](out)
                hidden_states = self.to_out[1](hidden_states)
                return hidden_states
            else:
                query = self.to_q(hidden_states)
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
                assert self._slice_size is None or query.shape[0] // self._slice_size == 1
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
                out = editor(q=query, k=key, v=value, is_cross=is_cross, place_in_unet=place_in_unet, num_heads=self.heads, attention_mask=attention_mask, scale=self.scale, sim=attention_scores, attn=attention_probs)
                hidden_states = self.to_out[0](out)
                hidden_states = self.to_out[1](hidden_states)
                return hidden_states

        return forward

    # def register_editor(net, count, place_in_unet):
    #     for name, subnet in net.named_children():
    #         print(net.__class__.__name__)
    #         if net.__class__.__name__ == 'SpatialTemporalAttention':  # spatial Transformer layer
    #             print("We are in the SpatialTemporalAttention condition")
    #             net.forward = ca_forward(net, place_in_unet)
    #             print("The forward is connected")
    #             return count + 1
    #         elif hasattr(net, 'children'):
    #             count = register_editor(subnet, count, place_in_unet)
    #     return count

    def register_editor(net_, count, place_in_unet):
        if net_.__class__.__name__ == 'SparseCausalAttention' or net_.__class__.__name__ == 'CrossAttention':
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



def regiter_mutual_attention_editor_diffusers_with_adapter(model, editor: MutualAttentionBase):
    """
    Register a attention editor to Diffuser Pipeline, refer from [Prompt-to-Prompt]
    """
    def ca_forward(self, place_in_unet):
        def forward(hidden_states, encoder_hidden_states=None, attention_mask=None, video_length=None, normal_infer=False, source_masks=None, target_masks=None, rectangle_source_masks=None):

            batch_size, sequence_length, _ = hidden_states.shape
            is_cross = encoder_hidden_states is not None
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

            assert self._slice_size is None or query.shape[0] // self._slice_size == 1
            if self.upcast_attention:
                query = query.float()
                key = key.float()

            out = editor(q=query, k=key, v=value, is_cross=is_cross, place_in_unet=place_in_unet, num_heads=self.heads, attention_mask=attention_mask, scale=self.scale)
            hidden_states = self.to_out[0](out)
            hidden_states = self.to_out[1](hidden_states)
            return hidden_states

        return forward

    def register_editor(net_, count, place_in_unet):
        if net_.__class__.__name__ == 'SparseCausalAttention' or net_.__class__.__name__ == 'CrossAttention':
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
