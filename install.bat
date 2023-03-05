rem Part of DiffusionMagic - Script to install DiffusionMagic
%~dp0tools\windows\micromamba.exe -r %~dp0env create -y -f %~dp0environment.yml
echo DiffusionMagic installation completed.
pause