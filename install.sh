#Part of DiffusionMagic - Script to install DiffusionMagic
BASEDIR=$(dirname "$0")
echo "$BASEDIR"
$BASEDIRtools/linux/micromamba -r $BASEDIRenv create -y -f $BASEDIRenvironment.yml
pause