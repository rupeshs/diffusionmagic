import torch
import gradio as gr
from computing import Computing
from stablediffusion.text_to_image import StableDiffusion
from stablediffusion.samplers  import Sampler
from stablediffusion.setting import StableDiffusionSetting,StableDiffusionImageToImageSetting

# model_id = "dreamlike-art/dreamlike-diffusion-1.0"
# spipeline = get_text_to_image_pipleline()
compute = Computing()
stable_diffusion = StableDiffusion(compute)
stable_diffusion.get_text_to_image_pipleline()


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
    vae_slicing,
    seed,
):
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
        vae_slicing=vae_slicing,
    )
    images = stable_diffusion.image_to_image(stable_diffusion_image_settings)
    return images

with gr.Blocks() as diffusion_magic:
    gr.Label("DiffusionMagic")
    with gr.Tabs():
        with gr.TabItem("Text to image"):
            with gr.Blocks() as sd_text_to_image:
                random_enabled = True
                with gr.Row():
                    with gr.Column():
                        def random_seed():
                            global random_enabled
                            random_enabled = not random_enabled
                            seed_val = -1
                            if not random_enabled:
                                seed_val = 42

                            return gr.Number.update(interactive= not random_enabled,value=seed_val)

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
                        seed = gr.Number(label="Seed", value=-1, precision=0,interactive= False)
                        seed_checkbox = gr.Checkbox(label="Use random seed",value=True,interactive=True)

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
        with gr.TabItem("Image to image"):
            with gr.Blocks() as sd_text_to_image:
                random_enabled = True
                with gr.Row():
                    with gr.Column():
                        def random_seed():
                            global random_enabled
                            random_enabled = not random_enabled
                            seed_val = -1
                            if not random_enabled:
                                seed_val = 42

                            return gr.Number.update(interactive=not random_enabled,value=seed_val)
                        input_image = gr.Image(label="Input image",type="pil")
                        strength = gr.Slider(
                                0.0, 1.0, value=0.7, step=0.05, label="Strength"
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

                        # vae_slicing = gr.Checkbox(
                        #         label="VAE slicing  (Enable if low VRAM)", value=True
                        #     )
                        seed = gr.Number(label="Seed", value=-1, precision=0,interactive= False)
                        seed_checkbox = gr.Checkbox(label="Use random seed",value=True,interactive=True)

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
                                vae_slicing,
                                seed,
                            ]

                    with gr.Column():
                        output = gr.Gallery(
                                label="Generated images", show_label=True, elem_id="gallery"
                            )
                generate_btn.click(
        fn=diffusion_image_to_image,
        inputs=input_params,
        outputs=output,
    )
                seed_checkbox.change(
        fn=random_seed,
        outputs=seed
      
    )


                


if __name__ == "__main__":
    diffusion_magic.launch()
