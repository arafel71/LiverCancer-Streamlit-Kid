#!/bin/bash

#--- Note:  this file is designed to run locally and within docker to prep the environment
#--- Entry:  this script is assumed to run from the /app root folder
#--- Usage:  ./scripts/util_local_runStreamlitFastApi.sh
echo -e "INFO(util_local_runStreamlitFastApi):\t Initializing ..."

#--- for volume initialization; ensure folders are in place; assume:  we are in the /app folder
mkdir -p data/demo_tiles/raw
mkdir -p data/tiles/raw data/tiles/pred data/tiles/grad_bg data/tiles/grad_wt data/tiles/grad_vt
mkdir -p data/wsi/raw

#--- for streamlit;  external 49400;  internal 39400
echo "INFO:  starting streamlit ..."
streamlit run app.py --server.port=39400 --server.maxUploadSize=2000 &

#--- for fastapi;  external 49500;  internal 39500
echo "INFO:  starting fastapi ..."
uvicorn main:app --reload --workers 1 --host 0.0.0.0 --port 39500 &

#--- wait for any process to exit
wait -n

#--- Exit with status of process that exited first
exit $?