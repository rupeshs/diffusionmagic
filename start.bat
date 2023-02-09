@echo off
rem Part of DiffusionMagic - Script to Start DiffusionMagic
echo Starting DiffusionMagic please wait...
cd %~dp0env\condabin\
call activate.bat
micromamba activate %~dp0env\envs\diffusionmagic-env && python %~dp0src\app.py

