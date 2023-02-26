#!/bin/bash
# Part of DiffusionMagic - Script to install DiffusionMagic
BASEDIR=$(pwd)
chmod +x $BASEDIR/tools/linux/micromamba
chmod +x start.sh
$BASEDIR/tools/linux/micromamba -r $BASEDIR/env create -y -f $BASEDIR/environment.yml
$BASEDIR/tools/linux/micromamba shell init -s bash -p $BASEDIR/umamba
read -n1 -r -p "Press any key to continue..." key
