#!/bin/bash

<<blkHeader
Name:       util_joinModel
Purpose:    reconstitutes a split pyTorch binary model with weights, into a single binary file
Usage:      ./util_joinModel.sh <source pattern match> <dest model file>
            - the first arg has to be wrapped in single quotes to ensure that bash does not expand wildcards
Prereqs:    a model folder within bin/models;  containing a split pyTorch model.pth as 1 or more model_nn files 
Todo:       get the parent folder name and use this as the name for the model file
blkHeader

#--- dependencies
#none


#--- initialize/configuration
#---    $1:  first arg;  source pattern match;  eg './bin/models/deeplabv3*vhflip30/model_a*';  Note that this is wildcarded so must be in quotes
#---    $n:  last arg; dest model file;  eg. ./bin/models/model.pth
echo -e "INFO(util_joinModel):\t Initializing ..."
strPth_patternMatch=$1
if [ -z "$strPth_patternMatch" ]; then
    echo "WARN: no args provided.  Exiting script."
    exit
fi

strPth_filMatch=( $strPth_patternMatch )        #--- expand the pattern match; get the first value of the pattern match
strPth_parentFld=$(dirname $strPth_filMatch)    #--- get the parent dir of the first file match
strPth_mdlFile=${@: -1}                         #--- Note:  this gets the last arg;  otherwise the 2nd arg would be an iteration of the 1st arg wildcard

strpth_pwd=$(pwd)
strpth_scriptLoc=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
strpth_scrHome="${strpth_scriptLoc}/../"
strpth_appHome="${strpth_scrHome}/../"

#echo "TRACE:  strPth_patternMatch= $strPth_patternMatch"
#echo "TRACE:  strPth_filMatch= $strPth_filMatch"
#echo "TRACE:  strPth_parentFld= $strPth_parentFld"
#echo "TRACE:  strPth_mdlFile= $strPth_mdlFile"

#--- reconstitute model
#--- Note:  cat command does not work with single-quote literals; do not reapply single quotes
#echo "cat ${strPth_patternMatch} > ${strPth_mdlFile}"
echo -e "INFO:\t Joining model binary ..."
cat ${strPth_patternMatch} > ${strPth_mdlFile}
echo -e "INFO:\t Done ...\n"