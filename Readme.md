## DiffusionMagic  Beta
StableDiffusion workflows using Diffusers. 

 ![ DiffusionMagic](https://raw.githubusercontent.com/rupeshs/diffusionmagic/main/docs/images/diffusion_magic.PNG)
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

## System Requirements:
- Works on Windows/Linux/Mac 64-bit
- Works on CPU,GPU,Apple Silicon M1/M2 hardware
- 12 GB System RAM
~11 GB disk space after installation (on SSD for best performance)
## Screenshots
## Image Inpainting
 ![ Image Inpainting](https://raw.githubusercontent.com/rupeshs/diffusionmagic/main/docs/images/diffusion_magic_inpainting.PNG)
## Depth to Image
 ![ Depth To Image](https://raw.githubusercontent.com/rupeshs/diffusionmagic/main/docs/images/diffusion_magic_depth_image.PNG)
 ## Instruction based image editing
 ![ Depth To Image](https://raw.githubusercontent.com/rupeshs/diffusionmagic/main/docs/images/diffusion_magic_instruct_to_pix.PNG
)



### How to install and run on Windows
Follow the steps to install and run the Diffusion magic on Windows.
- First we need to run the `install.bat` batch file it will install the necessary dependencies for DiffusionMagic.
(It will take some time to install,depends on your internet speed)
- Run the  `install.bat` script.
- To start DiffusionMagic click `start.bat`
### How to install and run on Linux
Follow the steps to install and run the Diffusion magic on Linux.

 - Run the following command:
  `chmod +x install.sh`
- Run the  `install.sh` script.
 ` ./install.sh`
- To start DiffusionMagic run:
` ./start.sh`

### How to install and run on Mac (Not tested)
Ensure the following prerequisites.
#### prerequisites 
- Mac computer with Apple silicon (M1/M2) hardware.
- macOS 12.6 or later (13.0 or later recommended).

Follow the steps to install and run the Diffusion magic on Mac (Apple Silicon M1/M2).
 - Run the following command:
  `chmod +x install-mac.sh`
- Run the  `install-mac.sh` script.
`./install-mac.sh`
- To start DiffusionMagic run:
` ./start.sh`

 Open the browser `http://localhost:7860/`
 Dark theme `http://localhost:7860/?__theme=dark`

## How to switch models
Diffusion magic will chnage UI based on the model selected.
Follow the steps to switch the models() inpainting,depth to image or instruct pix to pix or any other hugging face stable diffusion model)
- Start the Diffusion Magic app, open the settings tab and change the model
 ![ DiffusionMagic settings page](https://raw.githubusercontent.com/rupeshs/diffusionmagic/main/docs/images/diffusion_magic%20setting.PNG)
- Save the settings
- Close the app and start using start.bat/start.sh
 ### How to add new model
You can add new models hugging face model by 
- Adding huggugg id to the configs/stable_diffusion_models.txt file
- Adding locally copied model path to configs/stable_diffusion_models.txt file

