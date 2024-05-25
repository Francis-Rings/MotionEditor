import os
import torch
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from torchvision.utils import save_image

from motion_editor.attn_control.fully_control_utils import MutualAttentionBase, reshape_batch_dim_to_heads_base
from diffusers.utils.import_utils import is_xformers_available


if is_xformers_available():
    import xformers
    import xformers.ops
else:
    print(1/0)
    xformers = None

class MutualSelfAttentionControl(MutualAttentionBase):
    MODEL_TYPE = {
        "SD": 16,
        "SDXL": 70
    }

    def __init__(self, start_step=4, start_layer=10, layer_idx=None, step_idx=None, total_steps=50, model_type="SD"):
        """
        Mutual self-attention control for Stable-Diffusion model
        Args:
            start_step: the step to start mutual self-attention control
            start_layer: the layer to start mutual self-attention control
            layer_idx: list of the layers to apply mutual self-attention control
            step_idx: list the steps to apply mutual self-attention control
            total_steps: the total number of steps
            model_type: the model type, SD or SDXL
        """
        super().__init__()
        self.total_steps = total_steps
        self.total_layers = self.MODEL_TYPE.get(model_type, 16)
        self.start_step = start_step
        self.start_layer = start_layer
        self.layer_idx = layer_idx if layer_idx is not None else list(range(start_layer, self.total_layers))
        self.step_idx = step_idx if step_idx is not None else list(range(start_step, total_steps))
        print("denoising steps: ", self.step_idx)
        print("U-Net layers: ", self.layer_idx)

    def attn_batch(self, q, k, v, num_heads, sim=None, attn=None, is_cross=None, place_in_unet=None, **kwargs):
        """
        Performing attention for a batch of queries, keys, and values
        """
        b = q.shape[0] // num_heads
        q = rearrange(q, "(b h) n d -> h (b n) d", h=num_heads)
        k = rearrange(k, "(b h) n d -> h (b n) d", h=num_heads)
        v = rearrange(v, "(b h) n d -> h (b n) d", h=num_heads)

        sim = torch.einsum("h i d, h j d -> h i j", q, k) * kwargs.get("scale")
        attn = sim.softmax(-1)
        out = torch.einsum("h i j, h j d -> h i d", attn, v)
        out = rearrange(out, "h (b n) d -> b n (h d)", b=b)
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

        if is_cross or self.cur_step not in self.step_idx or self.cur_att_layer // 2 not in self.layer_idx:
            return super().forward(q=q, k=k, v=v, is_cross=is_cross, place_in_unet=place_in_unet, num_heads=num_heads, sim=sim, attn=attn, **kwargs)

        qu, qc = q.chunk(2)
        ku, kc = k.chunk(2)
        vu, vc = v.chunk(2)
        num_frames = q.size()[0] // num_heads // 4

        out_u_s = self.attn_batch(q=qu[:num_heads * num_frames], k=ku[:num_heads * num_frames], v=vu[:num_heads * num_frames], num_heads=num_heads,**kwargs)
        out_u_t = self.attn_batch(q=qu[num_heads * num_frames:], k=ku[:num_heads * num_frames], v=vu[:num_heads * num_frames], num_heads=num_heads,**kwargs)
        out_c_s = self.attn_batch(q=qc[:num_heads * num_frames], k=kc[:num_heads * num_frames], v=vc[:num_heads * num_frames], num_heads=num_heads,**kwargs)
        out_c_t = self.attn_batch(q=qc[num_heads * num_frames:], k=kc[:num_heads * num_frames], v=vc[:num_heads * num_frames], num_heads=num_heads,**kwargs)
        out = torch.cat([out_u_s, out_u_t, out_c_s, out_c_t], dim=0)

        return out


class MutualSelfAttentionControlUnion(MutualSelfAttentionControl):
    def __init__(self, start_step=4, start_layer=10, layer_idx=None, step_idx=None, total_steps=50, model_type="SD"):
        """
        Mutual self-attention control for Stable-Diffusion model with unition source and target [K, V]
        Args:
            start_step: the step to start mutual self-attention control
            start_layer: the layer to start mutual self-attention control
            layer_idx: list of the layers to apply mutual self-attention control
            step_idx: list the steps to apply mutual self-attention control
            total_steps: the total number of steps
            model_type: the model type, SD or SDXL
        """
        super().__init__(start_step, start_layer, layer_idx, step_idx, total_steps, model_type)

    def forward(self, q, k, v, sim=None, attn=None, is_cross=None, place_in_unet=None, num_heads=None, attention_mask=None, **kwargs):
        """
        Attention forward function
        """
        if is_cross or self.cur_step not in self.step_idx or self.cur_att_layer // 2 not in self.layer_idx:
            return super().forward(q=q, k=k, v=v, is_cross=is_cross, place_in_unet=place_in_unet, num_heads=num_heads, **kwargs)

        qu_s, qu_t, qc_s, qc_t = q.chunk(4)
        ku_s, ku_t, kc_s, kc_t = k.chunk(4)
        vu_s, vu_t, vc_s, vc_t = v.chunk(4)
        # attnu_s, attnu_t, attnc_s, attnc_t = attn.chunk(4)

        # source image branch
        out_u_s = super().forward(q=qu_s, k=ku_s, v=vu_s, is_cross=is_cross, place_in_unet=place_in_unet, num_heads=num_heads, **kwargs)
        out_c_s = super().forward(q=qc_s, k=kc_s, v=vc_s, is_cross=is_cross, place_in_unet=place_in_unet, num_heads=num_heads, **kwargs)

        # target image branch, concatenating source and target [K, V]
        out_u_t = self.attn_batch(q=qu_t, k=torch.cat([ku_s, ku_t]), v=torch.cat([vu_s, vu_t]), is_cross=is_cross, place_in_unet=place_in_unet, num_heads=num_heads, **kwargs)
        out_c_t = self.attn_batch(q=qc_t, k=torch.cat([kc_s, kc_t]), v=torch.cat([vc_s, vc_t]), is_cross=is_cross, place_in_unet=place_in_unet, num_heads=num_heads, **kwargs)

        out = torch.cat([out_u_s, out_u_t, out_c_s, out_c_t], dim=0)

        return out


class FullySelfAttentionControlMaskAuto(MutualSelfAttentionControl):
    def __init__(self, start_step=4, start_layer=10, layer_idx=None, step_idx=None, total_steps=50, thres=0.1, ref_token_idx=[1], cur_token_idx=[1], mask_save_dir=None, model_type="SD", source_masks=None, target_masks=None, rectangle_source_masks=None):
        """
        Args:
            start_step: the step to start mutual self-attention control
            start_layer: the layer to start mutual self-attention control
            layer_idx: list of the layers to apply mutual self-attention control
            step_idx: list the steps to apply mutual self-attention control
            total_steps: the total number of steps
            thres: the thereshold for mask thresholding
            ref_token_idx: the token index list for cross-attention map aggregation
            cur_token_idx: the token index list for cross-attention map aggregation
            mask_save_dir: the path to save the mask image
        """
        super().__init__(start_step, start_layer, layer_idx, step_idx, total_steps, model_type)
        print("Using MutualSelfAttentionControlMaskAuto")
        self.thres = thres
        self.ref_token_idx = ref_token_idx
        self.cur_token_idx = cur_token_idx

        self.self_attns = []
        self.cross_attns = []

        self.cross_attns_mask = None
        self.self_attns_mask = None

        self.mask_save_dir = mask_save_dir
        if self.mask_save_dir is not None:
            os.makedirs(self.mask_save_dir, exist_ok=True)

        self.source_masks = None
        self.target_masks = None
        self.rectangle_source_masks = None
        if source_masks is not None:
            self.source_masks = source_masks
            self.source_masks = rearrange(self.source_masks, "b f c h w -> b c f h w")
        if target_masks is not None:
            self.target_masks = target_masks
            self.target_masks = rearrange(self.target_masks, "b f c h w -> b c f h w")
        if rectangle_source_masks is not None:
            self.rectangle_source_masks = rectangle_source_masks
            self.rectangle_source_masks = rearrange(self.rectangle_source_masks, "b f c h w -> b c f h w")

    def after_step(self):
        self.self_attns = []
        self.cross_attns = []

    def attn_batch(self, q, k, v, num_heads, sim=None, attn=None, is_cross=None, place_in_unet=None, **kwargs):
        """
        Performing attention for a batch of queries, keys, and values
        """

        num_frames = 8
        H = W = int(np.sqrt(q.shape[1]))

        if self.source_masks is not None and kwargs.get("is_mask_attn"):
            # binarize the mask
            masks = self.source_masks
            masks = F.interpolate(masks, (num_frames, H, W), mode="nearest")
            curr_frame_index = torch.arange(num_frames)
            former_frame_index = torch.arange(num_frames) - 1
            former_frame_index[0] = 0
            masks = F.interpolate(masks, (num_frames, H, W), mode="nearest")
            prev_masks = masks[:, :, former_frame_index]
            cur_masks = masks[:, :, curr_frame_index]
            k_bg = k
            k_fg = k
            k_bg_prev = k_bg[:, :H*W]
            k_bg_cur = k_bg[:, H*W:]
            k_bg_prev = rearrange(k_bg_prev, "(b f) (h w) c -> b c f h w", f=num_frames, h=H, w=W)
            k_bg_prev = k_bg_prev * (1-prev_masks)
            k_bg_prev = rearrange(k_bg_prev, "b c f h w -> (b f) (h w) c", f=num_frames, h=H, w=W)
            k_bg_cur = rearrange(k_bg_cur, "(b f) (h w) c -> b c f h w", f=num_frames, h=H, w=W)
            k_bg_cur = k_bg_cur * (1-cur_masks)
            k_bg_cur = rearrange(k_bg_cur, "b c f h w -> (b f) (h w) c", f=num_frames, h=H, w=W)
            k_bg = torch.cat([k_bg_prev, k_bg_cur], dim=1)

            k_fg_prev = k_fg[:, :H * W]
            k_fg_cur = k_fg[:, H * W:]
            k_fg_prev = rearrange(k_fg_prev, "(b f) (h w) c -> b c f h w", f=num_frames, h=H, w=W)
            k_fg_prev = k_fg_prev * prev_masks
            k_fg_prev = rearrange(k_fg_prev, "b c f h w -> (b f) (h w) c", f=num_frames, h=H, w=W)
            k_fg_cur = rearrange(k_fg_cur, "(b f) (h w) c -> b c f h w", f=num_frames, h=H, w=W)
            k_fg_cur = k_fg_cur * cur_masks
            k_fg_cur = rearrange(k_fg_cur, "b c f h w -> (b f) (h w) c", f=num_frames, h=H, w=W)
            k_fg = torch.cat([k_fg_prev, k_fg_cur], dim=1)

            v_bg = v
            v_fg = v
            v_bg_prev = v_bg[:, :H * W]
            v_bg_cur = v_bg[:, H * W:]
            v_bg_prev = rearrange(v_bg_prev, "(b f) (h w) c -> b c f h w", f=num_frames, h=H, w=W)
            v_bg_prev = v_bg_prev * (1 - prev_masks)
            v_bg_prev = rearrange(v_bg_prev, "b c f h w -> (b f) (h w) c", f=num_frames, h=H, w=W)
            v_bg_cur = rearrange(v_bg_cur, "(b f) (h w) c -> b c f h w", f=num_frames, h=H, w=W)
            v_bg_cur = v_bg_cur * (1 - cur_masks)
            v_bg_cur = rearrange(v_bg_cur, "b c f h w -> (b f) (h w) c", f=num_frames, h=H, w=W)
            v_bg = torch.cat([v_bg_prev, v_bg_cur], dim=1)

            v_fg_prev = v_fg[:, :H * W]
            v_fg_cur = v_fg[:, H * W:]
            v_fg_prev = rearrange(v_fg_prev, "(b f) (h w) c -> b c f h w", f=num_frames, h=H, w=W)
            v_fg_prev = v_fg_prev * prev_masks
            v_fg_prev = rearrange(v_fg_prev, "b c f h w -> (b f) (h w) c", f=num_frames, h=H, w=W)
            v_fg_cur = rearrange(v_fg_cur, "(b f) (h w) c -> b c f h w", f=num_frames, h=H, w=W)
            v_fg_cur = v_fg_cur * cur_masks
            v_fg_cur = rearrange(v_fg_cur, "b c f h w -> (b f) (h w) c", f=num_frames, h=H, w=W)
            v_fg = torch.cat([v_fg_prev, v_fg_cur], dim=1)

            k = torch.cat([k_fg, k_bg], dim=0)
            q = torch.cat([q]*2)
            v = torch.cat([v_fg, v_bg], dim=0)

        attention_scores = torch.baddbmm(
            torch.empty(q.shape[0], q.shape[1], k.shape[1], dtype=q.dtype, device=q.device),
            q,
            k.transpose(-1, -2),
            beta=0,
            alpha=kwargs.get("scale"),
        )

        attention_probs = attention_scores.softmax(dim=-1)
        hidden_states = torch.bmm(attention_probs, v)
        out = reshape_batch_dim_to_heads_base(hidden_states, num_heads)
        return out

    def aggregate_cross_attn_map(self, idx):
        attn_map = torch.stack(self.cross_attns, dim=1).mean(1)  # (B, N, dim)
        B = attn_map.shape[0]
        res = int(np.sqrt(attn_map.shape[-2]))
        attn_map = attn_map.reshape(-1, res, res, attn_map.shape[-1])
        image = attn_map[..., idx]
        if isinstance(idx, list):
            image = image.sum(-1)
        image_min = image.min(dim=1, keepdim=True)[0].min(dim=2, keepdim=True)[0]
        image_max = image.max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0]
        image = (image - image_min) / (image_max - image_min)
        return image

    def forward(self, q, k, v, sim=None, attn=None, is_cross=None, place_in_unet=None, num_heads=None, attention_mask=None, **kwargs):
        """
        Attention forward function
        """

        if is_cross:
            if attn.shape[1] == 16 * 16:
                self.cross_attns.append(attn.reshape(-1, num_heads, *attn.shape[-2:]).mean(1))

        if is_cross or self.cur_step not in self.step_idx or self.cur_att_layer // 2 not in self.layer_idx:
            return super().forward(q=q, k=k, v=v, is_cross=is_cross, place_in_unet=place_in_unet, num_heads=num_heads, sim=sim, attn=attn, **kwargs)

        B = q.shape[0] // num_heads // 2
        H = W = int(np.sqrt(q.shape[1]))
        num_frames = q.size()[0] // num_heads // 4
        qu, qc = q.chunk(2)
        ku, kc = k.chunk(2)
        vu, vc = v.chunk(2)
        # attnu, attnc = attn.chunk(2)

        out_u_source = self.attn_batch(q=qu[:num_heads*num_frames], k=ku[:num_heads*num_frames], v=vu[:num_heads*num_frames], is_cross=is_cross, place_in_unet=place_in_unet, num_heads=num_heads, is_mask_attn=False, **kwargs)
        out_c_source = self.attn_batch(q=qc[:num_heads*num_frames], k=kc[:num_heads*num_frames], v=vc[:num_heads*num_frames], is_cross=is_cross, place_in_unet=place_in_unet, num_heads=num_heads, is_mask_attn=False, **kwargs)

        if len(self.cross_attns) == 0:
            self.self_attns_mask = None
            out_u_target = self.attn_batch(q=qu[-num_heads*num_frames:], k=ku[:num_heads*num_frames], v=vu[:num_heads*num_frames], is_cross=is_cross, place_in_unet=place_in_unet, num_heads=num_heads, is_mask_attn=False, **kwargs)
            out_c_target = self.attn_batch(q=qc[-num_heads*num_frames:], k=kc[:num_heads*num_frames], v=vc[:num_heads*num_frames], is_cross=is_cross, place_in_unet=place_in_unet, num_heads=num_heads, is_mask_attn=False, **kwargs)
        else:
            self.self_attns_mask = self.source_masks
            out_u_target = self.attn_batch(q=qu[-num_heads*num_frames:], k=ku[:num_heads*num_frames], v=vu[:num_heads*num_frames], is_cross=is_cross, place_in_unet=place_in_unet, num_heads=num_heads, is_mask_attn=True, **kwargs)
            out_c_target = self.attn_batch(q=qc[-num_heads*num_frames:], k=kc[:num_heads*num_frames], v=vc[:num_heads*num_frames], is_cross=is_cross, place_in_unet=place_in_unet, num_heads=num_heads, is_mask_attn=True, **kwargs)

        if self.self_attns_mask is not None:
            mask = self.aggregate_cross_attn_map(idx=self.cur_token_idx)  # (2, H, W)
            batch_frames, h_dim, w_dim = mask.size()
            num_frames = batch_frames // 4
            mask_target = mask[num_frames*3:]  # (H, W)
            res = int(np.sqrt(q.shape[1]))
            mask_target_tmp = []
            for i in range(mask_target.size()[0]):
                element = mask_target[i]
                element = F.interpolate(element.unsqueeze(0).unsqueeze(0), (res, res)).reshape(-1, 1)
                mask_target_tmp.append(element)
            spatial_mask = torch.stack(mask_target_tmp)

            # binarize the mask
            thres = self.thres
            spatial_mask[spatial_mask >= thres] = 1
            spatial_mask[spatial_mask < thres] = 0
            out_u_target_fg, out_u_target_bg = out_u_target.chunk(2)
            out_c_target_fg, out_c_target_bg = out_c_target.chunk(2)
            out_u_target = out_u_target_fg * spatial_mask + out_u_target_bg * (1 - spatial_mask)
            out_c_target = out_c_target_fg * spatial_mask + out_c_target_bg * (1 - spatial_mask)

            # set self self-attention mask to None
            self.self_attns_mask = None

        out = torch.cat([out_u_source, out_u_target, out_c_source, out_c_target], dim=0)
        return out


class FullySelfAttentionControlMask(MutualSelfAttentionControl):
    def __init__(self, start_step=4, start_layer=10, layer_idx=None, step_idx=None, total_steps=50, thres=0.1,
                 ref_token_idx=[1], cur_token_idx=[1], mask_save_dir=None, model_type="SD", source_masks=None,
                 target_masks=None, rectangle_source_masks=None):
        """
        Args:
            start_step: the step to start mutual self-attention control
            start_layer: the layer to start mutual self-attention control
            layer_idx: list of the layers to apply mutual self-attention control
            step_idx: list the steps to apply mutual self-attention control
            total_steps: the total number of steps
            thres: the thereshold for mask thresholding
            ref_token_idx: the token index list for cross-attention map aggregation
            cur_token_idx: the token index list for cross-attention map aggregation
            mask_save_dir: the path to save the mask image
        """
        super().__init__(start_step, start_layer, layer_idx, step_idx, total_steps, model_type)
        print("Using FullySelfAttentionControlMask")
        self.thres = thres
        self.ref_token_idx = ref_token_idx
        self.cur_token_idx = cur_token_idx

        self.self_attns = []
        self.cross_attns = []

        self.cross_attns_mask = None
        self.self_attns_mask = None

        self.mask_save_dir = mask_save_dir
        if self.mask_save_dir is not None:
            os.makedirs(self.mask_save_dir, exist_ok=True)

        self.source_masks = None
        self.target_masks = None
        self.rectangle_source_masks = None
        if source_masks is not None:
            self.source_masks = source_masks
            self.source_masks = rearrange(self.source_masks, "b f c h w -> b c f h w")
        else:
            print(1/0)

    def attn_batch(self, q, k, v, num_heads, sim=None, attn=None, is_cross=None, place_in_unet=None, attention_mask=None, **kwargs):
        """
        Performing attention for a batch of queries, keys, and values
        """

        num_frames = 8
        H = W = int(np.sqrt(q.shape[1]))

        if self.source_masks is not None and kwargs.get("is_mask_attn"):
            k_s_fg = k[:, :H*W*2]
            k_s_bg = k[:, :H*W*2]
            k_t = k[:, H * W * 3:]
            masks = self.source_masks
            curr_frame_index = torch.arange(num_frames)
            former_frame_index = torch.arange(num_frames) - 1
            former_frame_index[0] = 0
            masks = F.interpolate(masks, (num_frames, H, W), mode="nearest")
            masks_prev = masks[:, :, former_frame_index]
            masks_cur = masks[:, :, curr_frame_index]
            k_s_fg_prev = k_s_fg[:, :H*W]
            k_s_fg_cur = k_s_fg[:, H*W:]
            k_s_fg_prev = rearrange(k_s_fg_prev, "(b f) (h w) c -> b c f h w", f=num_frames, h=H, w=W)
            k_s_fg_prev = k_s_fg_prev * masks_prev
            k_s_fg_prev = rearrange(k_s_fg_prev, "b c f h w -> (b f) (h w) c", f=num_frames, h=H, w=W)
            k_s_fg_cur = rearrange(k_s_fg_cur, "(b f) (h w) c -> b c f h w", f=num_frames, h=H, w=W)
            k_s_fg_cur = k_s_fg_cur * masks_cur
            k_s_fg_cur = rearrange(k_s_fg_cur, "b c f h w -> (b f) (h w) c", f=num_frames, h=H, w=W)
            k_s_fg = torch.cat([k_s_fg_prev, k_s_fg_cur], dim=1)
            k_s_bg_prev = k_s_bg[:, :H * W]
            k_s_bg_cur = k_s_bg[:, H * W:]
            k_s_bg_prev = rearrange(k_s_bg_prev, "(b f) (h w) c -> b c f h w", f=num_frames, h=H, w=W)
            k_s_bg_prev = k_s_bg_prev * (1-masks_prev)
            k_s_bg_prev = rearrange(k_s_bg_prev, "b c f h w -> (b f) (h w) c", f=num_frames, h=H, w=W)
            k_s_bg_cur = rearrange(k_s_bg_cur, "(b f) (h w) c -> b c f h w", f=num_frames, h=H, w=W)
            k_s_bg_cur = k_s_bg_cur * (1-masks_cur)
            k_s_bg_cur = rearrange(k_s_bg_cur, "b c f h w -> (b f) (h w) c", f=num_frames, h=H, w=W)
            k_s_bg = torch.cat([k_s_bg_prev, k_s_bg_cur], dim=1)
            k = torch.cat([k_s_fg, k_s_bg, k_t], dim=1)
            v_s_fg = v[:, :H * W * 2]
            v_s_bg = v[:, :H * W * 2]
            v_t = v[:, H * W * 3:]
            v = torch.cat([v_s_fg, v_s_bg, v_t], dim=1)

        query = q.contiguous()
        key = k.contiguous()
        value = v.contiguous()
        out = xformers.ops.memory_efficient_attention(query, key, value, attn_bias=attention_mask)

        hidden_states = out.to(query.dtype)
        out = reshape_batch_dim_to_heads_base(hidden_states, num_heads)
        return out


    def forward(self, q, k, v, sim=None, attn=None, is_cross=None, place_in_unet=None, num_heads=None, attention_mask=None, **kwargs):
        """
        Attention forward function
        """

        if is_cross:
            if attn.shape[1] == 16 * 16:
                self.cross_attns.append(attn.reshape(-1, num_heads, *attn.shape[-2:]).mean(1))

        if is_cross or self.cur_step not in self.step_idx or self.cur_att_layer // 2 not in self.layer_idx:
            return super().forward(q=q, k=k, v=v, is_cross=is_cross, place_in_unet=place_in_unet, num_heads=num_heads,
                                   sim=sim, attn=attn, **kwargs)

        B = q.shape[0] // num_heads // 2
        H = W = int(np.sqrt(q.shape[1]))
        num_frames = q.size()[0] // num_heads // 4
        qu_s, qu_t, qc_s, qc_t = q.chunk(4)
        ku_s, ku_t, kc_s, kc_t = k.chunk(4)
        vu_s, vu_t, vc_s, vc_t = v.chunk(4)
        out_u_source = self.attn_batch(q=qu_s, k=ku_s, v=vu_s, is_cross=is_cross, place_in_unet=place_in_unet,num_heads=num_heads, is_mask_attn=False, attention_mask=attention_mask, **kwargs)
        out_c_source = self.attn_batch(q=qc_s, k=kc_s, v=vc_s, is_cross=is_cross, place_in_unet=place_in_unet,num_heads=num_heads, is_mask_attn=False, attention_mask=attention_mask,**kwargs)
        out_u_target = self.attn_batch(q=qu_t, k=torch.cat([ku_s, ku_t], dim=1), v=torch.cat([vu_s, vu_t], dim=1), is_cross=is_cross,place_in_unet=place_in_unet, num_heads=num_heads, is_mask_attn=True, cur_step=self.cur_step, attention_mask=attention_mask, **kwargs)
        out_c_target = self.attn_batch(q=qc_t, k=torch.cat([kc_s, kc_t], dim=1), v=torch.cat([vc_s, vc_t], dim=1), is_cross=is_cross,place_in_unet=place_in_unet, num_heads=num_heads, is_mask_attn=True, cur_step=self.cur_step, attention_mask=attention_mask, **kwargs)

        if self.target_masks is not None and self.source_masks is not None:
            out_u_target_fg, out_u_target_bg = out_u_target.chunk(2)
            out_c_target_fg, out_c_target_bg = out_c_target.chunk(2)
            masks = self.target_masks
            masks = F.interpolate(masks, (num_frames, H, W), mode="nearest")
            masks = torch.squeeze(masks).flatten(1)
            masks = torch.unsqueeze(masks, dim=-1)
            out_u_target = out_u_target_fg * masks + out_u_target_bg * (1 - masks)
            out_c_target = out_c_target_fg * masks + out_c_target_bg * (1 - masks)

        out = torch.cat([out_u_source, out_u_target, out_c_source, out_c_target], dim=0)
        return out


