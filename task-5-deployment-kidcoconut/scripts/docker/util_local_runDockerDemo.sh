#!/bin/bash

#--- Note:  this file is designed to run locally to launch docker
#--- Entry:  this script is assumed to run from the /app root folder
#--- Usage:  ./scripts/util_local_runDockerDemo.sh
#--- Assume:  docker image has been built;  container is not running

<<blockComment 
    util_local_runDockerDemo -> Dockerfile -> util_dockerPreRun -> util_local_runStreamlitFastApi
blockComment


#--- initialize/config
kstr_defDkrHubId="kidcoconut73"
kstr_defDkrImageName="img_stm_omdenasaudi_hcc"
kstr_defDkrCtrName=${kstr_defDkrImageName/img_/ctr_}
kstr_defDkrTagVersion="0.1.2"
kstr_defDkrTagStage="demo"

kstr_dkrImg="${kstr_defDkrImageName}:${kstr_defDkrTagVersion}"
kstr_dkrCtr="${kstr_defDkrImageName/img_/ctr_}"          #--- bash replace one occurrence

#--- stop the container if it is running
docker stop $kstr_dkrCtr

#--- delete container if it exists
docker rm $kstr_dkrCtr

#--- to run the container from the image;  specific port mapping (-p) vs any available port mapping (-P)
#    docker run -p 49400:39400 -p 49500:39500 --name ctr_stmOmdenaSaudiHcc -v ./data:/app/data img_stm_omdenasaudi_hcc:0.1

#--- run docker demo locally
docker run -p 49400:39400 -p 49500:39500 --name $kstr_dkrCtr -v ./data:/app/data $kstr_dkrImg
