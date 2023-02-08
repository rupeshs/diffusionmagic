import torch
import gradio as gr
from computing import Computing
from stablediffusion.text_to_image import get_text_to_image_pipleline

computing = Computing()

# model_id = "dreamlike-art/dreamlike-diffusion-1.0"
pipeline = get_text_to_image_pipleline()


def diffusion_text_to_image(
    prompt,
    neg_prompt,
    image_height,
    image_width,
    inference_steps,
    guidance_scale,
    num_images,
    attention_slicing,
    vae_slicing,
    seed,
):
    generator = None
    if seed != -1:
        generator = torch.Generator(computing.name).manual_seed(seed)

    if attention_slicing:
        pipeline.enable_attention_slicing()
    else:
        pipeline.disable_attention_slicing()

    if vae_slicing:
        pipeline.enable_vae_slicing()
    else:
        pipeline.disable_vae_slicing()

    images = pipeline(
        prompt,
        guidance_scale=guidance_scale,
        num_inference_steps=inference_steps,
        height=image_height,
        width=image_width,
        negative_prompt=neg_prompt,
        num_images_per_prompt=num_images,
        generator=generator,
    ).images

    return images


with gr.Blocks() as sd_text_to_image:
    gr.Label("DiffusionMagic")
    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(
                label="Describe the image you'd like to see",
                lines=3,
                placeholder="A fantasy landscape",
            )
            neg_prompt = gr.Textbox(
                label="Don't want to see",
                lines=1,
                placeholder="",
                value="bad, deformed, ugly, bad Anatomy",
            )
            image_height = gr.Slider(
                512, 2048, value=512, step=64, label="Image Height"
            )
            image_width = gr.Slider(512, 2048, value=512, step=64, label="Image Width")
            num_inference_steps = gr.Slider(
                1, 100, value=20, step=1, label="Inference Steps"
            )
            guidance_scale = gr.Slider(
                1.0, 30.0, value=7.5, step=0.5, label="Guidance Scale"
            )
            num_images = gr.Slider(
                1, 50, value=1, step=1, label="Number of images to generate"
            )
            attn_slicing = gr.Checkbox(
                label="Attention slicing (Enable if low VRAM)", value=True
            )

            vae_slicing = gr.Checkbox(
                label="VAE slicing  (Enable if low VRAM)", value=True
            )

            seed = gr.Number(label="Seed", value=-1, precision=0)

            input_params = [
                prompt,
                neg_prompt,
                image_height,
                image_width,
                num_inference_steps,
                guidance_scale,
                num_images,
                attn_slicing,
                vae_slicing,
                seed,
            ]
            print(input_params)
            generate_btn = gr.Button("Generate")

        with gr.Column():
            output = gr.Gallery(
                label="Generated images", show_label=True, elem_id="gallery"
            )

    generate_btn.click(
        fn=diffusion_text_to_image,
        inputs=input_params,
        outputs=output,
    )

if __name__ == "__main__":
    sd_text_to_image.launch()
