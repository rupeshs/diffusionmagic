## DiffusionMagic  Beta
DiffusionMagic is simple to use Stable Diffusion workflows using [diffusers](https://github.com/huggingface/diffusers). 
DiffusionMagic focused on the following areas:
- Easy to use
- Cross-platform (Windows/Linux/Mac)
- Modular design, latest best optimizations for speed and memory

 ![ DiffusionMagic](https://raw.githubusercontent.com/rupeshs/diffusionmagic/main/docs/images/diffusion_magic.PNG)
## Features
- Supports various Stable Diffusion workflows
    - Text to Image 
    - Image to Image 
    - Image variations
    - Image Inpainting
    - Depth to Image
    - Instruction based image editing
- Supports Controlnet workflows
   - Canny
   - MLSD (Line control)
   -  Normal
   - HED
   - Pose
   - Depth
   - Scribble
   - Segmentation
- Pytorch 2.0 support
- Supports all stable diffusion Hugging Face models 
- Supports Stable diffusion v1 and v2 models, derived models
- Works on Windows/Linux/Mac 64-bit
- Works on CPU,GPU,Apple Silicon M1/M2 hardware
- Supports DEIS scheduler for faster image generation (10 steps)
- Supports 7 different samplers with latest DEIS sampler
- LoRA(Low-Rank Adaptation of Large Language Models) models support (~3 MB size)
- Easy to add new diffuser model by updating stable_diffusion_models.txt 
- Low VRAM mode supports GPU with RAM < 4 GB 
- Fast model loading
- Supports Attention slicing and VAE slicing
- Simple installation using install.bat/install.sh

## Screenshots
### Image variations
 ![  Image variations](https://raw.githubusercontent.com/rupeshs/diffusionmagic/main/docs/images/diffusion_magic_image_variations.PNG)
### Image Inpainting
 ![ Image Inpainting](https://raw.githubusercontent.com/rupeshs/diffusionmagic/main/docs/images/diffusion_magic_inpainting.PNG)
### Depth to Image
 ![ Depth To Image](https://raw.githubusercontent.com/rupeshs/diffusionmagic/main/docs/images/diffusion_magic_depth_image.PNG)
 ### Instruction based image editing
 ![ Depth To Image](https://raw.githubusercontent.com/rupeshs/diffusionmagic/main/docs/images/diffusion_magic_instruct_to_pix.PNG
)
## System Requirements:
- Works on Windows/Linux/Mac 64-bit
- Works on CPU,GPU,Apple Silicon M1/M2 hardware
- 12 GB System RAM
- ~11 GB disk space after installation (on SSD for best performance)

## Download Release
Download release from the github DiffusionMagic releases.
## How to install and run on Windows
Follow the steps to install and run the Diffusion magic on Windows.
- First we need to run(double click) the `install.bat` batch file it will install the necessary dependencies for DiffusionMagic.
(It will take some time to install,depends on your internet speed)
- Run the  `install.bat` script.
- To start DiffusionMagic double click `start.bat`


 ![ DiffusionMagic started on Windows](https://raw.githubusercontent.com/rupeshs/diffusionmagic/main/docs/images/diffusion_magic_windows.PNG)
## How to install and run on Linux
Follow the steps to install and run the Diffusion magic on Linux.

 - Run the following command:
  `chmod +x install.sh`
- Run the  `install.sh` script.
 ` ./install.sh`
- To start DiffusionMagic run:
` ./start.sh`

## How to install and run on Mac (Not tested)
*Testers needed - If you have MacOS feel free to test and contribute*

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

## Colab Support
Yes, you can run DiffusionMagic in Colab.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/116wcgjsvLhw9IrHeBDlvPQSW2oVzsnql?usp=sharing)
## How to switch models
Diffusion magic will change UI based on the model selected.
Follow the steps to switch the models() inpainting,depth to image or instruct pix to pix or any other hugging face stable diffusion model)
- Start the Diffusion Magic app, open the settings tab and change the model
 ![ DiffusionMagic settings page](https://raw.githubusercontent.com/rupeshs/diffusionmagic/main/docs/images/diffusion_magic%20setting.PNG)
- Save the settings
- Close the app and start using start.bat/start.sh
 ## How to add new model
We can add any Hugging Face stable diffusion model to DiffusionMagic by 
- Adding Hugging Face models  id or local folder path to the configs/stable_diffusion_models.txt file
E.g `https://huggingface.co/dreamlike-art/dreamlike-diffusion-1.0`
Here model id is `dreamlike-art/dreamlike-diffusion-1.0`
Or we can clone the model use the local folder path as model id.
- Adding locally copied model path to configs/stable_diffusion_models.txt file
## Linting (Development)
Run the following commands from src folder
`mypy --ignore-missing-imports --explicit-package-bases .`
`flake8 --max-line-length=100 .`
## Contribute
Contributions are welcomed.

