@echo off
rem Part of DiffusionMagic - Script to Start DiffusionMagic
echo Starting DiffusionMagic please wait...
call %~dp0env\condabin\activate.bat
micromamba activate %~dp0env\envs\diffusionmagic-env && python %~dp0src\app.py
