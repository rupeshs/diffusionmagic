from typing import Any

import gradio as gr

from backend.stablediffusion.models.scheduler_types import (
    SchedulerType,
    get_sampler_names,
)

random_enabled = True


def get_image_to_image_ui(generate_callback_fn: Any) -> None:
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

                input_image = gr.Image(label="Input image", type="pil")
                strength = gr.Slider(
                    0.0,
                    1.0,
                    value=0.75,
                    step=0.05,
                    label="Strength",
                )

                prompt = gr.Textbox(
                    label="Describe the image you'd like to see",
                    lines=3,
                    placeholder="A fantasy landscape",
                )

                neg_prompt = gr.Textbox(
                    label="Don't want to see",
                    lines=1,
                    placeholder="",
                    value="bad, deformed, ugly, bad anatomy",
                )
                with gr.Accordion("Advanced options", open=False):
                    image_height = gr.Slider(
                        512, 2048, value=512, step=64, label="Image Height"
                    )
                    image_width = gr.Slider(
                        512, 2048, value=512, step=64, label="Image Width"
                    )
                    num_inference_steps = gr.Slider(
                        1, 100, value=20, step=1, label="Inference Steps"
                    )
                    scheduler = gr.Dropdown(
                        get_sampler_names(),
                        value=SchedulerType.DPMSolverMultistepScheduler.value,
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
                    )
                    attn_slicing = gr.Checkbox(
                        label="Attention slicing (Enable if low VRAM)",
                        value=True,
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

                input_params = [
                    input_image,
                    strength,
                    prompt,
                    neg_prompt,
                    image_height,
                    image_width,
                    num_inference_steps,
                    scheduler,
                    guidance_scale,
                    num_images,
                    attn_slicing,
                    seed,
                ]

            with gr.Column():
                generate_btn = gr.Button("Generate", elem_id="generate_button")
                output = gr.Gallery(
                    label="Generated images",
                    show_label=True,
                    elem_id="gallery",
                ).style(
                    grid=2,
                )
        generate_btn.click(
            fn=generate_callback_fn,
            inputs=input_params,
            outputs=output,
        )
        seed_checkbox.change(fn=random_seed, outputs=seed)
