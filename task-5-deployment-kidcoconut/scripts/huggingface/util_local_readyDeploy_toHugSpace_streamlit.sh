#!/bin/bash

#--- Note:  this file is designed to run locally to ready the deploy branch for hugspace
#--- Entry:  this script is assumed to run from the /app root folder
#--- Usage:  ./scripts/util_local_readyDeploy_toHugSpace_streamlit.sh

<<blockComment 
    bash:   util_local_readyDeploy_toHugSpace_streamlit -> git
    git:    local/task-5-deployment -> omdena/deploy_hugspace_streamlit -> hugspace/main
blockComment


#--- initialize/configuration
echo "TRACE:  Initializing ..."
kstr_hugspaceId="kidcoconut"


#--- git checkout deploy_hugspace_streamlit
#--- git merge task-5-deployment
#--- delete all unnecessary files
<<deadCode
    kstr_defDkrHubId="kidcoconut73"
    kstr_defDkrImageName="img_stm_omdenasaudi_hcc"
    kstr_defDkrTagVersion="0.1.2"
    kstr_defDkrTagStage="demo"
    strpth_pwd=$(pwd)
    strpth_scriptLoc=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )


    #--- declarations
    echo "TRACE:  Declarations ..."

    #strUtl_scriptLoc="$(utl_getScriptLoc)"
    source ${strpth_scriptLoc}/util.sh

    #kstr_dkrImg="kidcoconut73/img_stm_omdenasaudi_hcc:demo"
    #kstr_dkrCtr="kidcoconut73/ctr_stm_omdenasaudi_hcc:demo"
    kstr_dkrHubImg="${kstr_defDkrHubId}/${kstr_defDkrImageName}:${kstr_defDkrTagStage}"
    kstr_dkrImg="${kstr_defDkrImageName}:${kstr_defDkrTagVersion}"
    kstr_dkrCtr="${kstr_dkrImg/img_/ctr_}"          #--- bash replace one occurrence



    function utl_trace_config () {
        echo ""
        utl_trace_var "strpth_pwd" $strpth_pwd
        utl_trace_var "strpth_scriptLoc" $strpth_scriptLoc
        echo ""
        utl_trace_var "kstr_defDkrHubId" $kstr_defDkrHubId
        utl_trace_var "kstr_defDkrImageName" $kstr_defDkrImageName
        utl_trace_var "kstr_defDkrTagVersion" $kstr_defDkrTagVersion
        utl_trace_var "kstr_defDkrTagStage" $kstr_defDkrTagStage
        echo ""
        utl_trace_var "kstr_dkrHubImg" $kstr_dkrHubImg
        utl_trace_var "kstr_dkrImg" $kstr_dkrImg
        utl_trace_var "kstr_dkrCtr" $kstr_dkrCtr
        echo ""
    }

    echo -e "\nTRACE:  Echo config ...\n"
    utl_trace_config


    #--- to build/rebuild the image;  make sure you stop and remove the container if you are replacing/upgrading;  or change the version tag# from 0.1
    #--- stop the container if it is running
    #--- delete container if it exists
    echo -e "\nTRACE:  Stop and remove container if it exists ..."
    docker stop $kstr_dkrCtr
    docker rm $kstr_dkrCtr

    #--- build the docker image
    echo -e "\nTRACE:  Build the docker image ..."
    docker build -t $kstr_dkrImg .


    #--- to tag the image prior to push to DockerHub;  docker login and then register user/image:tag
    #--- to push this image to DockerHub, example based on the repo:  kidcoconut73/img_stm_omdenasaudi_hcc
    #    docker tag img_omdenasaudi_hcc:0.1 kidcoconut73/img_stm_omdenasaudi_hcc:demo
    #    docker tag img_omdenasaudi_hcc:0.1 kidcoconut73/img_stm_omdenasaudi_hcc:0.1
    #--- tag the image
    echo -e "\nTRACE:  Tag the image ..."
    docker tag ${kstr_dkrImg} $kstr_dkrHubImg
    docker tag ${kstr_dkrImg} "${kstr_defDkrHubId}/${kstr_defDkrImageName}:${kstr_defDkrTagVersion}"


    #--- push the image to dockerHub
    #    docker push kidcoconut73/img_stm_omdenasaudi_hcc:demo
deadCode