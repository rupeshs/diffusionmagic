from typing import Any

import gradio as gr

from backend.stablediffusion.models.scheduler_types import (
    SchedulerType,
    get_sampler_names,
)

random_enabled = True


def get_illusion_diffusion_to_image_ui(generate_callback_fn: Any) -> None:
    with gr.Blocks():
        with gr.Row():
            with gr.Column():

                def random_seed():
                    global random_enabled
                    random_enabled = not random_enabled
                    seed_val = -1
                    if not random_enabled:
                        seed_val = 42

                    return gr.Number.update(
                        interactive=not random_enabled, value=seed_val
                    )

                input_image = gr.Image(
                    label="Input image", type="pil", elem_id="control_image"
                )

                prompt = gr.Textbox(
                    label="Describe the image you'd like to see",
                    lines=3,
                    placeholder="A beautiful sunset",
                )

                neg_prompt = gr.Textbox(
                    label="Don't want to see",
                    lines=1,
                    placeholder="",
                    value="low quality,bad, deformed, ugly, bad anatomy",
                )
                controlnet_conditioning_scale = gr.Slider(
                    0.0,
                    5.0,
                    value=0.8,
                    step=0.01,
                    label="Illusion strength",
                )
                gr.Examples(
                    examples=[
                        "spiral.jpeg",
                        "diffusion_text.jpg",
                        "women.jpg",
                    ],
                    inputs=input_image,
                )
                with gr.Accordion("Advanced options", open=False):
                    image_height = gr.Slider(
                        512,
                        2048,
                        value=512,
                        step=64,
                        label="Image Height",
                        visible=False,
                    )
                    image_width = gr.Slider(
                        512,
                        2048,
                        value=512,
                        step=64,
                        label="Image Width",
                        visible=False,
                    )
                    num_inference_steps = gr.Slider(
                        2,
                        100,
                        value=20,
                        step=1,
                        label="Inference Steps",
                    )
                    scheduler = gr.Dropdown(
                        get_sampler_names(),
                        value=SchedulerType.EulerAncestralDiscreteScheduler.value,
                        label="Sampler",
                    )
                    guidance_scale = gr.Slider(
                        1.0,
                        30.0,
                        value=7.5,
                        step=0.5,
                        label="Guidance Scale",
                    )
                    num_images = gr.Slider(
                        1,
                        50,
                        value=1,
                        step=1,
                        label="Number of images to generate",
                        visible=False,
                    )

                    seed = gr.Number(
                        label="Seed",
                        value=-1,
                        precision=0,
                        interactive=False,
                    )
                    seed_checkbox = gr.Checkbox(
                        label="Use random seed",
                        value=True,
                        interactive=True,
                    )

                    control_guidance_start = gr.Slider(
                        0.0,
                        1.0,
                        value=0.0,
                        step=0.1,
                        label="Controlnet guidance start",
                    )
                    control_guidance_end = gr.Slider(
                        0.0,
                        1.0,
                        value=1.0,
                        step=0.1,
                        label="Controlnet guidance end",
                    )
                    upscaler_strength = gr.Slider(
                        0.0,
                        1.0,
                        value=1.0,
                        step=0.1,
                        label="Upscaler strength",
                        visible=False,
                    )

                input_params = [
                    input_image,
                    prompt,
                    neg_prompt,
                    image_height,
                    image_width,
                    num_inference_steps,
                    guidance_scale,
                    num_images,
                    scheduler,
                    seed,
                    controlnet_conditioning_scale,
                    control_guidance_start,
                    control_guidance_end,
                    upscaler_strength,
                ]

            with gr.Column():
                generate_btn = gr.Button(
                    "Generate illusion Image", elem_id="generate_button"
                )
                output = gr.Gallery(
                    label="Generated images",
                    show_label=True,
                    elem_id="gallery",
                    show_share_button=True,
                ).style(
                    grid=2,
                )
        generate_btn.click(
            fn=generate_callback_fn,
            inputs=input_params,
            outputs=output,
        )
        seed_checkbox.change(fn=random_seed, outputs=seed)
