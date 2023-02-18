from typing import Any

import gradio as gr

from backend.computing import Computing
from backend.stablediffusion.setting import (
    StableDiffusionImageToImageSetting,
    StableDiffusionSetting,
    StableDiffusionImageInpaintingSetting,
)
from backend.stablediffusion.stablediffusion import StableDiffusion
from backend.stablediffusion.inpainting import StableDiffusionInpainting
from frontend.web.image_to_image_ui import get_image_to_image_ui
from frontend.web.settings_ui import get_settings_ui
from frontend.web.text_to_image_ui import get_text_to_image_ui
from models.settings import DiffusionMagicSettings
from utils import DiffusionMagicPaths
from frontend.web.image_inpainting import get_image_inpainting_ui

compute = Computing()
stable_diffusion = StableDiffusion(compute)
stable_diffusion_inpainting = StableDiffusionInpainting(compute)


def diffusion_text_to_image(
    prompt,
    neg_prompt,
    image_height,
    image_width,
    inference_steps,
    scheduler,
    guidance_scale,
    num_images,
    attention_slicing,
    vae_slicing,
    seed,
) -> Any:
    stable_diffusion_settings = StableDiffusionSetting(
        prompt=prompt,
        negative_prompt=neg_prompt,
        image_height=image_height,
        image_width=image_width,
        inference_steps=inference_steps,
        guidance_scale=guidance_scale,
        number_of_images=num_images,
        scheduler=scheduler,
        seed=seed,
        attention_slicing=attention_slicing,
        vae_slicing=vae_slicing,
    )
    images = stable_diffusion.text_to_image(stable_diffusion_settings)
    return images


def diffusion_image_to_image(
    image,
    strength,
    prompt,
    neg_prompt,
    image_height,
    image_width,
    inference_steps,
    scheduler,
    guidance_scale,
    num_images,
    attention_slicing,
    seed,
) -> Any:
    stable_diffusion_image_settings = StableDiffusionImageToImageSetting(
        image=image,
        strength=strength,
        prompt=prompt,
        negative_prompt=neg_prompt,
        image_height=image_height,
        image_width=image_width,
        inference_steps=inference_steps,
        guidance_scale=guidance_scale,
        number_of_images=num_images,
        scheduler=scheduler,
        seed=seed,
        attention_slicing=attention_slicing,
    )
    images = stable_diffusion.image_to_image(stable_diffusion_image_settings)
    return images


def diffusion_image_inpainting(
    image,
    prompt,
    neg_prompt,
    image_height,
    image_width,
    inference_steps,
    scheduler,
    guidance_scale,
    num_images,
    attention_slicing,
    seed,
) -> Any:
    stable_diffusion_image_settings = StableDiffusionImageInpaintingSetting(
        image=image["image"],
        mask_image=image["mask"],
        prompt=prompt,
        negative_prompt=neg_prompt,
        image_height=image_height,
        image_width=image_width,
        inference_steps=inference_steps,
        guidance_scale=guidance_scale,
        number_of_images=num_images,
        scheduler=scheduler,
        seed=seed,
        attention_slicing=attention_slicing,
    )
    images = stable_diffusion_inpainting.image_inpainting(
        stable_diffusion_image_settings
    )
    return images


def diffusionmagic_web_ui(settings: DiffusionMagicSettings) -> gr.Blocks:
    model_id = settings.model_settings.model_id
    is_inpainting = False
    if "inpainting" in model_id:
        is_inpainting = True
        stable_diffusion_inpainting.get_inpainting_pipleline(model_id)
    else:
        stable_diffusion.get_text_to_image_pipleline(model_id)

    with gr.Blocks(
        css=DiffusionMagicPaths.get_css_path(),
        title="DiffusionMagic",
    ) as diffusion_magic_ui:
        gr.HTML("<center><H3>DiffusionMagic</H3></center>")
        with gr.Tabs():
            if not is_inpainting:
                with gr.TabItem("Text to image"):
                    get_text_to_image_ui(diffusion_text_to_image)
                with gr.TabItem("Image to image"):
                    get_image_to_image_ui(diffusion_image_to_image)
            else:
                with gr.TabItem("Image Inpainting"):
                    get_image_inpainting_ui(diffusion_image_inpainting)
            with gr.TabItem("Settings"):
                get_settings_ui()
    return diffusion_magic_ui
