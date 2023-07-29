#!/bin/bash

#--- Note:  this file is designed to run locally as well as within docker to prep the environment
#--- Entry:  this script is assumed to run from the /app root folder
#--- Usage:  ./scripts/docker/util_docker_preRun.sh

#--- for volume initialization; ensure folders are in place; assume:  we are in the /app folder


<<blockComment 
- the binary model is stored as split files named mdl_nn
- this is done to ensure that the model can be stored within gitHub
- the split model is recreated on docker container startup using the cat command
blockComment
echo -e "INFO(util_docker_preRun):\t Initializing ..."

strpth_pwd=$(pwd)
strpth_scriptLoc=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
strpth_scrHome="${strpth_scriptLoc}/../"
strpth_appHome="${strpth_scrHome}../"
strpth_scrModels="${strpth_scrHome}models/"

echo "strpth_appHome = ${strpth_appHome}"

#--- recreate single model file from its parts, stored within a specific model version folder
strpth_binModels="${strpth_appHome}bin/models/"
echo "strpth_binModels = ${strpth_binModels}"
#$("'${strpth_scrModels}util_joinModel.sh' '${strpth_binModels}deeplabv3*vhflip30/model_a*' '${strpth_binModels}model.pth'")
eval "'${strpth_scrModels}/util_joinModel.sh' '${strpth_binModels}/deeplabv3*vhflip30/model_a*' '${strpth_binModels}/model.pth'"

#--- run streamlit/fastapi
eval "'${strpth_scrHome}/streamlitFastApi/util_local_runStreamlitFastApi.sh'"