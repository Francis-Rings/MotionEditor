import os
import imageio
import tempfile
import numpy as np
from PIL import Image
from typing import Union

import torch
import torchvision

from tqdm import tqdm
from einops import rearrange


def save_videos_as_images(videos: torch.Tensor, path: str, rescale=False, n_rows=4, fps=1):
    # dir_name = os.path.dirname(path)
    dir_name = path
    videos = rearrange(videos, "b c t h w -> t b h w c")

    os.makedirs(os.path.join(dir_name, "vis_images"), exist_ok=True)
    for frame_idx, x in enumerate(videos):
        if rescale:
            x = (x + 1.0) / 2.0
        x = (x * 255).numpy().astype(np.uint8)

        for batch_idx, image in enumerate(x):
            save_dir = os.path.join(dir_name, "vis_images", f"batch_{batch_idx}")
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"frame_{frame_idx}.png")
            image = Image.fromarray(image)
            image.save(save_path)


def save_videos_grid(videos: torch.Tensor, path: str, rescale=False, n_rows=4, fps=1):
    videos = rearrange(videos, "b c t h w -> t b c h w")
    outputs = []
    for x in videos:
        x = torchvision.utils.make_grid(x, nrow=n_rows)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        x = (x * 255).numpy().astype(np.uint8)
        outputs.append(x)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.mimsave(path, outputs, fps=8)

    # save for gradio demo
    out_file = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
    out_file.name = path.replace('.gif', '.mp4')
    writer = imageio.get_writer(out_file.name, fps=fps)
    for frame in outputs:
        writer.append_data(frame)
    writer.close()


@torch.no_grad()
def init_prompt(prompt, pipeline):
    uncond_input = pipeline.tokenizer(
        [""], padding="max_length", max_length=pipeline.tokenizer.model_max_length,
        return_tensors="pt"
    )
    uncond_embeddings = pipeline.text_encoder(uncond_input.input_ids.to(pipeline.device))[0]
    text_input = pipeline.tokenizer(
        [prompt],
        padding="max_length",
        max_length=pipeline.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = pipeline.text_encoder(text_input.input_ids.to(pipeline.device))[0]
    context = torch.cat([uncond_embeddings, text_embeddings])

    return context


def next_step(model_output: Union[torch.FloatTensor, np.ndarray], timestep: int,
              sample: Union[torch.FloatTensor, np.ndarray], ddim_scheduler):
    timestep, next_timestep = min(
        timestep - ddim_scheduler.config.num_train_timesteps // ddim_scheduler.num_inference_steps, 999), timestep
    alpha_prod_t = ddim_scheduler.alphas_cumprod[timestep] if timestep >= 0 else ddim_scheduler.final_alpha_cumprod
    alpha_prod_t_next = ddim_scheduler.alphas_cumprod[next_timestep]
    beta_prod_t = 1 - alpha_prod_t
    next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
    next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
    next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
    return next_sample


def get_noise_pred_single(latents, t, context, unet, normal_infer=False, image_embeddings=None):
    bs = latents.shape[0]  # (b*f, c, h, w) or (b, c, f, h, w)
    if bs != context.shape[0]:
        context = context.repeat(bs, 1, 1)  # (b*f, len, dim)
    # noise_pred = unet(latents, t, encoder_hidden_states=context, normal_infer=normal_infer)["sample"]
    noise_pred = unet(latents, t, encoder_hidden_states=context, normal_infer=normal_infer, class_labels=image_embeddings)["sample"]
    return noise_pred

@torch.no_grad()
def init_image_embed(image_embeds, pipeline, noise_level, generator):
    dtype = next(pipeline.image_encoder.parameters()).dtype
    device = pipeline.image_encoder.device
    noise_level = torch.tensor([noise_level], device=device)
    image_embeds = pipeline.noise_image_embeddings(
        image_embeds=image_embeds,
        noise_level=noise_level,
        generator=generator,
    ) # 1,1024
    return image_embeds


@torch.no_grad()
def ddim_loop(pipeline, ddim_scheduler, latent, num_inv_steps, prompt, normal_infer=False, image_embed=None, noise_level=None, generator=None):
    context = init_prompt(prompt, pipeline)
    uncond_embeddings, cond_embeddings = context.chunk(2)

    all_latent = [latent]
    latent = latent.clone().detach()

    for i in tqdm(range(num_inv_steps)):
        t = ddim_scheduler.timesteps[len(ddim_scheduler.timesteps) - i - 1]
        noise_pred = get_noise_pred_single(latent, t, cond_embeddings, pipeline.unet, normal_infer=normal_infer, image_embeddings=None)
        latent = next_step(noise_pred, t, latent, ddim_scheduler)
        all_latent.append(latent)
    return all_latent


@torch.no_grad()
def ddim_inversion(pipeline, ddim_scheduler, video_latent, num_inv_steps, prompt="", normal_infer=False, image_embed=None, noise_level=0, generator=None):
    ddim_latents = ddim_loop(pipeline, ddim_scheduler, video_latent, num_inv_steps, prompt, normal_infer=normal_infer, image_embed=image_embed, noise_level=noise_level, generator=generator)
    return ddim_latents
