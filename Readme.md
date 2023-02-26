## DiffusionMagic  Beta
StableDiffusion workflows using Diffusers. 

### Features
- Supports various image generation methods 
    -  Text to Image 
    -  Image to Image 
    - Image Inpainting
    - Depth to Image
    - Instruct Pix to Pix generation
-  Latest Stable diffusion 2.1
- Easy to add new diffuser model by updating stable_diffusion_models.txt 
- Supports DEIS scheduler for faster image generation (10 steps)
- Low VRAM mode supports GPU with RAM < 4 GB 

### How to install on Windows
First we need to install necessary dependencies for diffusion magic it will take some time to install(depends on your internet speed)
 Run the  `install.bat` script
 
 To start DiffusionMagic run 
 `start.bat `

 Open the browser `http://localhost:7860/`
 Dark theme `http://localhost:7860/?__theme=dark`
<!-- ### How to add new model
You can add new models hugging face model by adding id to the configs/stable_diffusion_models.txt file. -->
