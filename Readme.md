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

## System Requirements:
- Works on Windows/Linux/Mac 64-bit
- Works on CPU,GPU,Apple Silicon M1/M2 hardware
- 12 GB System RAM
~11 GB disk space after installation (on SSD for best performance)
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
<!-- ### How to add new model
You can add new models hugging face model by adding id to the configs/stable_diffusion_models.txt file. -->
