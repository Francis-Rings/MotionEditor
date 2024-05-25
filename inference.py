import argparse
import datetime
import logging
import inspect
import math
import os
import warnings
from typing import Dict, Optional, Tuple
from omegaconf import OmegaConf

import torch
import torch.nn.functional as F
import torch.utils.checkpoint

import diffusers
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler, DDIMScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from motion_editor.attn_control.control_utils import regiter_mutual_attention_editor_diffusers
from motion_editor.attn_control.fully_control import FullySelfAttentionControlMask
from motion_editor.attn_control.fully_control_utils import regiter_fully_attention_editor_diffusers
from motion_editor.attn_control.temporal_control import TemporalSelfAttentionControl
from motion_editor.attn_control.temporal_control_utils import regiter_temporal_attention_editor_diffusers
from motion_editor.models.unet_2d_condition import UNet2DConditionModel
from motion_editor.data.dataset import VideoDataset
from motion_editor.p2p.null_text_optimization import MyNullInversion
from motion_editor.pipelines.pipeline_motion_editor import MotionEditorPipeline
from motion_editor.util import save_videos_grid, save_videos_as_images, ddim_inversion
from einops import rearrange
from diffusers import ControlNetModel
import numpy as np
import re

from motion_editor.p2p.p2p_stable import AttentionReplace, AttentionRefine
from motion_editor.p2p.ptp_utils import register_attention_control


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.10.0.dev0")

logger = get_logger(__name__, log_level="INFO")


def get_word_inds(text: str, word_place, tokenizer):
    split_text = text.split(" ")
    if type(word_place) is str:
        word_place = [i for i, word in enumerate(split_text) if word_place == word]
    elif type(word_place) is int:
        word_place = [word_place]
    out = []
    if len(word_place) > 0:
        words_encode = [tokenizer.decode([item]).strip("#") for item in tokenizer.encode(text)][1:-1]
        print(words_encode)
        cur_len, ptr = 0, 0

        for i in range(len(words_encode)):
            cur_len += len(words_encode[i])
            if ptr in word_place:
                out.append(i + 1)
            if cur_len >= len(split_text[ptr]):
                ptr += 1
                cur_len = 0
    return out


def prepare_control(unet, prompts, validation_data):
    assert len(prompts) == 2

    print(prompts[0])
    print(prompts[1])
    length1 = len(prompts[0].split(' '))
    length2 = len(prompts[1].split(' '))
    if length1 == length2:
        # prepare for attn guidance
        cross_replace_steps = 0.8
        self_replace_steps = 0.4
        controller = AttentionReplace(prompts, validation_data['num_inference_steps'], 
                                      cross_replace_steps=cross_replace_steps,
                                      self_replace_steps=self_replace_steps)
    else:
        cross_replace_steps = 0.8
        self_replace_steps = 0.4
        controller = AttentionRefine(prompts, validation_data['num_inference_steps'],
                                     cross_replace_steps=self_replace_steps, 
                                     self_replace_steps=self_replace_steps)

    print(controller)
    register_attention_control(unet, controller)

    # the update of unet forward function is inplace
    return cross_replace_steps, self_replace_steps


def main(
    pretrained_model_path: str,
    output_dir: str,
    input_data: Dict,
    validation_data: Dict,
    input_batch_size: int = 1,
    gradient_accumulation_steps: int = 1,
    gradient_checkpointing: bool = True,
    mixed_precision: Optional[str] = "fp16",
    enable_xformers_memory_efficient_attention: bool = True,
    seed: Optional[int] = None,
    use_sc_attn: bool = True,
    use_st_attn: bool = True,
    st_attn_idx: int = 0,
    fps: int = 8,
    resume_from_checkpoint: Optional[str] = None,
    adapter_weight_path: Optional[str] = None,
):
    *_, config = inspect.getargvalues(inspect.currentframe())

    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if seed is not None:
        set_seed(seed)

    # Handle the output folder creation
    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/sample", exist_ok=True)
        OmegaConf.save(config, os.path.join(output_dir, 'config.yaml'))

    # Load tokenizer and models.
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(pretrained_model_path, subfolder="unet", use_sc_attn=use_sc_attn, use_st_attn=use_st_attn, st_attn_idx=st_attn_idx)
    controlnet = ControlNetModel.from_pretrained("checkpoints/sd-controlnet-openpose", torch_dtype=torch.float16)

    # Freeze vae, text_encoder, and unet
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    controlnet.requires_grad_(False)

    if enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    if gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Get the training dataset
    input_dataset = VideoDataset(**input_data)

    # Preprocessing the dataset
    input_dataset.prompt_ids = tokenizer(
        input_dataset.prompt, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
    ).input_ids[0]

    # DataLoaders creation:
    input_dataloader = torch.utils.data.DataLoader(
        input_dataset, batch_size=input_batch_size
    )

    # Get the validation pipeline
    validation_pipeline = MotionEditorPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=DDIMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler"),
        safety_checker=None,
        feature_extractor=None,
        controlnet=controlnet,
    )
    validation_pipeline.enable_vae_slicing()
    ddim_inv_scheduler = DDIMScheduler.from_pretrained(pretrained_model_path, subfolder='scheduler')
    ddim_inv_scheduler.set_timesteps(validation_data.num_inv_steps)

    # Prepare everything with our `accelerator`.
    unet, input_dataloader = accelerator.prepare(
        unet, input_dataloader,
    )

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move text_encode and vae to gpu and cast to weight_dtype
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    controlnet.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(input_dataloader) / gradient_accumulation_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("vid2vid-zero")

    # Zero-shot Eval!
    total_batch_size = input_batch_size * accelerator.num_processes * gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(input_dataset)}")
    logger.info(f"  Instantaneous batch size per device = {input_batch_size}")
    logger.info(f"  Total input batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    global_step = 0


    accelerator.load_state(resume_from_checkpoint)
    controlnet_adapter_weight_path = adapter_weight_path
    controlnet_adapter_weight = torch.load(controlnet_adapter_weight_path)
    unet.controlnet_adapter.load_state_dict(controlnet_adapter_weight)

    for name, module in validation_pipeline.unet.named_modules():
        print(name)
    print("========================================")
    for name, params in validation_pipeline.unet.named_parameters():
        print(name)

    unet.eval()
    for step, batch in enumerate(input_dataloader):
        samples = []
        sample_reconstruct = []
        edited_sample_imgs = []
        reconstruct_sample_imgs = []
        pixel_values = batch["pixel_values"].to(weight_dtype)
        # save input video 
        video = (pixel_values / 2 + 0.5).clamp(0, 1).detach().cpu()
        video = video.permute(0, 2, 1, 3, 4)  # (b, f, c, h, w)
        samples.append(video)
        # start processing
        video_length = pixel_values.shape[1]
        pixel_values = rearrange(pixel_values, "b f c h w -> (b f) c h w")
        latents = vae.encode(pixel_values).latent_dist.sample()
        # take video as input
        latents = rearrange(latents, "(b f) c h w -> b c f h w", f=video_length)
        latents = latents * 0.18215

        source_skeleton = batch["source_conditions"]["openposefull"].to(weight_dtype)
        target_skeleton = batch["target_conditions"]["openposefull"].to(weight_dtype)

        source_masks = batch["source_masks"].to(weight_dtype)
        source_masks = source_masks.to(device=unet.device, dtype=unet.dtype)
        generator = torch.Generator(device="cuda")
        generator.manual_seed(seed)

        # perform inversion
        ddim_inv_latent = None
        if validation_data.use_null_inv:
            null_inversion = MyNullInversion(
                model=validation_pipeline, guidance_scale=validation_data.guidance_scale, null_inv_with_prompt=False,
                null_normal_infer=False,
            )
            ddim_inv_latent, uncond_embeddings = null_inversion.invert(
                latents, input_dataset.prompt,
                verbose=True,
            )
            ddim_inv_latent = ddim_inv_latent.to(weight_dtype)
            uncond_embeddings = [embed.to(weight_dtype) for embed in uncond_embeddings]
        else:
            ddim_inv_latent = ddim_inversion(
                validation_pipeline, ddim_inv_scheduler, video_latent=latents,
                num_inv_steps=validation_data.num_inv_steps, prompt="",
                normal_infer=True,  # we don't want to use scatn or denseattn for inversion, just use sd inferenece
            )[-1].to(weight_dtype)
            uncond_embeddings = None

        ddim_inv_latent = ddim_inv_latent.repeat(2, 1, 1, 1, 1)

        for idx, prompt in enumerate(validation_data.prompts):
            prompts = [input_dataset.prompt, prompt]  # a list of two prompts
            validation_target_skeleton = target_skeleton
            validation_source_skeleton = torch.zeros_like(validation_target_skeleton)
            skeleton = torch.cat([validation_source_skeleton, validation_target_skeleton, validation_source_skeleton, validation_target_skeleton], dim=0)

            train_index = get_word_inds(text=prompts[0], word_place="girl", tokenizer=tokenizer)
            validate_index = get_word_inds(text=prompts[1], word_place="girl", tokenizer=tokenizer)

            STEP = 4
            LAYPER = 10
            temporal_editor = TemporalSelfAttentionControl(start_step=STEP, start_layer=LAYPER)
            regiter_temporal_attention_editor_diffusers(validation_pipeline, temporal_editor)
            fully_editor = FullySelfAttentionControlMask(start_step=STEP, start_layer=LAYPER, ref_token_idx=train_index, cur_token_idx=validate_index, source_masks=source_masks, target_masks=None, rectangle_source_masks=None, mask_save_dir=None)
            regiter_fully_attention_editor_diffusers(validation_pipeline, fully_editor)

            sample = validation_pipeline(prompts,
                                         generator=generator,
                                         latents=ddim_inv_latent,
                                         uncond_embeddings=uncond_embeddings,
                                         skeleton=skeleton,
                                         source_masks=None,
                                         target_masks=None,
                                         rectangle_source_masks=None,
                                         background_latents=None,
                                         **validation_data).images

            assert sample.shape[0] == 2
            sample_inv, sample_gen = sample.chunk(2)
            # add input for vis
            save_videos_grid(sample_gen, f"{output_dir}/sample/{prompts[1]}.gif", fps=fps)
            save_videos_grid(sample_inv, f"{output_dir}/sample/{prompts[1]}-inv.gif", fps=fps)
            samples.append(sample_gen)
            sample_reconstruct.append(sample_inv)
            edited_sample_imgs.append(sample_gen)
            reconstruct_sample_imgs.append(sample_inv)

        samples = torch.concat(samples)
        sample_reconstruct = torch.concat(sample_reconstruct)
        save_path = f"{output_dir}/sample-all.gif"
        save_reconstruct_path = f"{output_dir}/sample-all-inv.gif"
        save_videos_grid(samples, save_path, fps=fps)
        save_videos_grid(samples, save_path.replace(".gif", ".mp4"), fps=fps)
        save_videos_grid(sample_reconstruct, save_reconstruct_path, fps=fps)
        save_videos_grid(sample_reconstruct, save_reconstruct_path.replace(".gif", ".mp4"), fps=fps)
        logger.info(f"Saved samples to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/motion_editor.yaml")
    args = parser.parse_args()

    main(**OmegaConf.load(args.config))

