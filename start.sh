# Part of DiffusionMagic - Script to Start DiffusionMagic
echo Starting DiffusionMagic please wait...
BASEDIR=$(dirname "$0")
cd $BASEDIRenv\condabin\
call activate.sh
micromamba activate %~dp0env\envs\diffusionmagic-env && python %~dp0src\app.py

