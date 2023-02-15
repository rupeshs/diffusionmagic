from typing import Any

import gradio as gr

from backend.computing import Computing
from backend.stablediffusion.setting import (
    StableDiffusionImageToImageSetting,
    StableDiffusionSetting,
)
from backend.stablediffusion.stablediffusion import StableDiffusion
from frontend.web.image_to_image_ui import get_image_to_image_ui
from frontend.web.settings_ui import get_settings_ui
from frontend.web.text_to_image_ui import get_text_to_image_ui
from models.settings import DiffusionMagicSettings
from utils import DiffusionMagicPaths

compute = Computing()
stable_diffusion = StableDiffusion(compute)


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


def diffusionmagic_web_ui(settings: DiffusionMagicSettings) -> gr.Blocks:
    model_id = settings.model_settings.model_id
    # stable_diffusion.get_text_to_image_pipleline(model_id)

    with gr.Blocks(
        css=DiffusionMagicPaths.get_css_path(),
        title="DiffusionMagic",
    ) as diffusion_magic_ui:
        gr.Label("DiffusionMagic")
        with gr.Tabs():
            with gr.TabItem("Text to image"):
                get_text_to_image_ui(diffusion_text_to_image)
            with gr.TabItem("Image to image"):
                get_image_to_image_ui(diffusion_image_to_image)
            with gr.TabItem("Settings"):
                get_settings_ui()
    return diffusion_magic_ui
