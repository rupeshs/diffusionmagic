from typing import Any

from backend.computing import Computing
from backend.imagesaver import ImageSaver
from backend.stablediffusion.depth_to_image import StableDiffusionDepthToImage
from backend.stablediffusion.inpainting import StableDiffusionInpainting
from backend.stablediffusion.instructpix import StableDiffusionInstructPixToPix
from backend.stablediffusion.models.setting import (
    StableDiffusionImageDepthToImageSetting,
    StableDiffusionImageInpaintingSetting,
    StableDiffusionImageToImageSetting,
    StableDiffusionSetting,
    StableDiffusionImageInstructPixToPixSetting,
    StableDiffusionControlnetSetting,
)
from backend.controlnet.ControlContext import ControlnetContext
from backend.stablediffusion.stablediffusion import StableDiffusion
from settings import AppSettings


class Generate:
    def __init__(self, compute: Computing):
        self.pipe_initialized = False
        self.inpaint_pipe_initialized = False
        self.depth_pipe_initialized = False
        self.pix_to_pix_initialized = False
        self.controlnet_initialized = False
        self.stable_diffusion = StableDiffusion(compute)
        self.stable_diffusion_inpainting = StableDiffusionInpainting(compute)
        self.stable_diffusion_depth = StableDiffusionDepthToImage(compute)
        self.stable_diffusion_pix_to_pix = StableDiffusionInstructPixToPix(compute)
        self.controlnet = ControlnetContext(compute)
        self.app_settings = AppSettings().get_settings()
        self.model_id = self.app_settings.model_settings.model_id
        self.low_vram_mode = self.app_settings.low_memory_mode

    def diffusion_text_to_image(
        self,
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
        self._init_stable_diffusion()
        images = self.stable_diffusion.text_to_image(stable_diffusion_settings)
        self._save_images(
            images,
            "TextToImage",
        )
        return images

    def _init_stable_diffusion(self):
        if not self.pipe_initialized:
            print("Initializing stable diffusion pipeline")
            self.stable_diffusion.get_text_to_image_pipleline(
                self.model_id,
                self.low_vram_mode,
            )
            self.pipe_initialized = True

    def diffusion_image_to_image(
        self,
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
        self._init_stable_diffusion()
        images = self.stable_diffusion.image_to_image(stable_diffusion_image_settings)

        self._save_images(
            images,
            "ImageToImage",
        )
        return images

    def diffusion_image_inpainting(
        self,
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

        if not self.inpaint_pipe_initialized:
            print("Initializing stable diffusion inpainting pipeline")
            self.stable_diffusion_inpainting.get_inpainting_pipleline(
                self.model_id,
                self.low_vram_mode,
            )
            self.inpaint_pipe_initialized = True

        images = self.stable_diffusion_inpainting.image_inpainting(
            stable_diffusion_image_settings
        )
        self._save_images(
            images,
            "Inpainting",
        )
        return images

    def diffusion_depth_to_image(
        self,
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
        stable_diffusion_image_settings = StableDiffusionImageDepthToImageSetting(
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

        if not self.depth_pipe_initialized:
            print("Initializing stable diffusion depth to image pipeline")

            self.stable_diffusion_depth.get_depth_to_image_pipleline(
                self.model_id,
                self.low_vram_mode,
            )
            self.depth_pipe_initialized = True
        images = self.stable_diffusion_depth.depth_to_image(
            stable_diffusion_image_settings
        )
        self._save_images(
            images,
            "DepthToImage",
        )
        return images

    def _save_images(
        self,
        images: Any,
        folder: str,
    ):
        if AppSettings().get_settings().output_images.use_seperate_folders:
            ImageSaver.save_images(
                AppSettings().get_settings().output_images.path,
                images,
                folder,
                AppSettings().get_settings().output_images.format,
            )
        else:
            ImageSaver.save_images(
                AppSettings().get_settings().output_images.path,
                images,
                "",
                AppSettings().get_settings().output_images.format,
            )

    def diffusion_pix_to_pix(
        self,
        image,
        image_guidance,
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
        stable_diffusion_image_settings = StableDiffusionImageInstructPixToPixSetting(
            image=image,
            image_guidance=image_guidance,
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
        if not self.pix_to_pix_initialized:
            print("Initializing stable diffusion instruct pix to pix pipeline")
            self.stable_diffusion_pix_to_pix.get_instruct_pix_to_pix_pipleline(
                self.model_id,
                self.low_vram_mode,
            )
            self.pix_to_pix_initialized = True

        images = self.stable_diffusion_pix_to_pix.instruct_pix_to_pix(
            stable_diffusion_image_settings
        )
        self._save_images(
            images,
            "InstructEditImage",
        )
        return images

    def diffusion_image_variations(
        self,
        image,
        strength,
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
            prompt="",
            negative_prompt="bad, deformed, ugly, bad anatomy",
            image_height=image_height,
            image_width=image_width,
            inference_steps=inference_steps,
            guidance_scale=guidance_scale,
            number_of_images=num_images,
            scheduler=scheduler,
            seed=seed,
            attention_slicing=attention_slicing,
        )
        self._init_stable_diffusion()
        images = self.stable_diffusion.image_to_image(stable_diffusion_image_settings)

        self._save_images(
            images,
            "ImageVariations",
        )
        return images

    def diffusion_control_to_image(
        self,
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
        vae_slicing,
        seed,
    ) -> Any:
        stable_diffusion_image_settings = StableDiffusionControlnetSetting(
            image=image,
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
        if not self.controlnet_initialized:
            print("Initializing controlnet image pipeline")
            self.controlnet.init_control_to_image_pipleline(
                model_id=self.model_id,
                low_vram_mode=self.low_vram_mode,
            )
            self.controlnet_initialized = True

        images = self.controlnet.control_to_image(stable_diffusion_image_settings)

        self._save_images(
            images,
            "CannyToImage",
        )
        return images
