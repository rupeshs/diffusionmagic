from typing import Any
import gradio as gr


random_enabled = True


def get_text_to_image_wuerstchen_ui(generate_callback_fn: Any) -> None:
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

                # with gr.Row():
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
                        1024, 2048, value=1024, step=64, label="Image Height"
                    )
                    image_width = gr.Slider(
                        1024, 2048, value=1536, step=64, label="Image Width"
                    )
                    prior_guidance_scale = gr.Slider(
                        1.0, 10.0, value=4.0, step=0.5, label="Prior guidance Scale"
                    )
                    num_images = gr.Slider(
                        1,
                        50,
                        value=1,
                        step=1,
                        label="Number of images to generate",
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
                        prompt,
                        neg_prompt,
                        image_height,
                        image_width,
                        prior_guidance_scale,
                        num_images,
                        seed,
                    ]

            with gr.Column():
                generate_btn = gr.Button("Generate", elem_id="generate_button")
                output = gr.Gallery(
                    label="Generated images",
                    show_label=True,
                    elem_id="gallery",
                    columns=2,
                )
    seed_checkbox.change(fn=random_seed, outputs=seed)
    generate_btn.click(
        fn=generate_callback_fn,
        inputs=input_params,
        outputs=output,
    )
