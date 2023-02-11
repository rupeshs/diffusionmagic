import torch
import gradio as gr
from computing import Computing
from stablediffusion.text_to_image import StableDiffusion
from stablediffusion.samplers  import Sampler
from stablediffusion.setting import StableDiffusionSetting

# model_id = "dreamlike-art/dreamlike-diffusion-1.0"
# spipeline = get_text_to_image_pipleline()
compute = Computing()
stable_diffusion = StableDiffusion(compute)
#stable_diffusion.get_text_to_image_pipleline()


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
):
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




with gr.Blocks() as sd_text_to_image:
    gr.Label("DiffusionMagic")

    random_enabled = True
    with gr.Row():
        with gr.Column():
            def random_seed():
                global random_enabled
                random_enabled = not random_enabled
                return gr.Number.update(interactive=random_enabled,value= -1)

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
            generate_btn = gr.Button("Generate")

            image_height = gr.Slider(
                512, 2048, value=512, step=64, label="Image Height"
            )
            image_width = gr.Slider(512, 2048, value=512, step=64, label="Image Width")
            num_inference_steps = gr.Slider(
                1, 100, value=20, step=1, label="Inference Steps"
            )
            samplers = [sampler.value for sampler in Sampler]
            scheduler = gr.Dropdown(
                samplers,
                value=Sampler.DPMSolverMultistepScheduler.value,
                label="Sampler",
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
            #with gr.Row():
            seed = gr.Number(label="Seed", value=-1, precision=0)
                #random_seed_btn = gr.Button("Random")
            seed_checkbox = gr.Checkbox(label="Use random seed")

            input_params = [
                prompt,
                neg_prompt,
                image_height,
                image_width,
                num_inference_steps,
                scheduler,
                guidance_scale,
                num_images,
                attn_slicing,
                vae_slicing,
                seed,
            ]

        with gr.Column():
            output = gr.Gallery(
                label="Generated images", show_label=True, elem_id="gallery"
            )

    generate_btn.click(
        fn=diffusion_text_to_image,
        inputs=input_params,
        outputs=output,
    )
    seed_checkbox.change(
        fn=random_seed,
        outputs=seed
      
    )

if __name__ == "__main__":
    sd_text_to_image.launch()
