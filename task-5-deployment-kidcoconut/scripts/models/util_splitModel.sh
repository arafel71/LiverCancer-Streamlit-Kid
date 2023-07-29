#!/bin/bash

<<blkHeader
Name:       util_splitModel
Purpose:    convenience script to split a single pyTorch .pth model file with weights into smaller 10MB chunks in order to store within github
Usage:      ./util_splitModel.sh <src model file> <dest folder>
            - the first arg has to be wrapped in single quotes to ensure that bash does not expand wildcards
Prereqs:    a pytorch model file 
Todo:       get the parent folder name and use this as the name for the model file
blkHeader

#--- dependencies
#none


#---    initialization/configuration
#---    $1:  first arg; the source model file; eg ./bin/models/model.pth
#---    $n:  last arg; dest model path;  eg. ./test_model_folder
strPth_mdlFile=$1
strPth_mdlFolder=$2
strPrefix='/model_'

if [ -z "$strPth_mdlFile" ] || [ -z "$strPth_mdlFolder" ]; then
    echo "WARN: no args provided.  Exiting script."
    exit
fi

strpth_pwd=$(pwd)
strpth_scriptLoc=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
strpth_scrHome="${strpth_scriptLoc}/../"
#strpth_ignHome="${strpth_scrHome}/../"
strpth_appHome="${strpth_scrHome}/../"

#echo "TRACE:  strPth_mdlFile= $strPth_mdlFile"
echo "TRACE:  strPth_mdlFolder= $strPth_mdlFolder"

#--- ensure the target dir exists
mkdir -p $strPth_mdlFolder

#--- split the model into smaller chunks
echo "split -b 10M $strPth_mdlFile $strPth_mdlFolder$strPrefix"
split -b 10M $strPth_mdlFile $strPth_mdlFolder$strPrefix

echo -e "INFO:\t Done ...\n"