#--- PREREQS:  
#   - create a local folder dedicated to WSI image mgmt:  (docker pwd)/data 
#   - populate the folder with raw data, wsi and tiles
#   - docker run --name <name> -v <local folder>

#--- utilize a light linux distro for python apps
FROM python:3.10.9-slim-bullseye

#--- copy only the requirements.txt file
#--- set docker image working directory to /app
#--- Not:  this is reorg'd in an attempt to reduce the rebuilding of layers
COPY ./requirements.txt /app/requirements.txt

#--- set docker image working directory to /app
WORKDIR /app

#--- install all lib dependencies into the image
RUN pip3 install -r ./requirements.txt
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

#--- copy all files from the local pwd to the docker image /app folder
#--- .dockerignore:  ensure no local data folders or files (images) are copied into the docker image/container 
COPY . /app

#--- for streamlit;  external 49400;  internal 39400
#       localExec:  (from root folder) streamlit run app.py --server.port=39400 --server.maxUploadSize=2000
EXPOSE 49400         
#CMD ["streamlit", "run", "app.py", "--server.port=39400", "--server.maxUploadSize=2000"]


#--- for fastapi;  external 49500;  internal 39500
#       localExec:  (from root folder) uvicorn main:app --reload --workers 1 --host 0.0.0.0 --port 39500 
EXPOSE 49500        
#CMD ["uvicorn", "main:app", "--reload", "--host=0.0.0.0", "--port=39500"]

#--- establish environment prereqs
#--- start streamlit and fastapi from a helper utility script
CMD ./scripts/docker/util_docker_preRun.sh


#--- to build/rebuild the image;  make sure you stop and remove the container if you are replacing/upgrading;  or change the version tag# from 0.1
#    docker build -t img_stm_omdenasaudi_hcc:0.1 .

#--- to tag the image prior to push to DockerHub;  docker login and then register user/image:tag
#--- to push this image to DockerHub, example based on the repo:  kidcoconut73/img_stm_omdenasaudi_hcc
#    docker tag img_omdenasaudi_hcc:0.1 kidcoconut73/img_stm_omdenasaudi_hcc:demo
#    docker tag img_omdenasaudi_hcc:0.1 kidcoconut73/img_stm_omdenasaudi_hcc:0.1
#    docker push kidcoconut73/img_stm_omdenasaudi_hcc:demo

#--- to run the container from the image;  specific port mapping (-p) vs any available port mapping (-P)
#    docker run -p 49400:39400 -p 49500:39500 --name ctr_stmOmdenaSaudiHcc -v ./data:/app/data img_stm_omdenasaudi_hcc:0.1
#    docker run -p 49400:39400 -p 49500:39500 --name ctr_stmOmdenaSaudiHcc img_stm_omdenasaudi_hcc:0.1
#    docker run -P --name ctr_stmOmdenaHcc img_stm_omdenasaudi_hcc:0.1                   #--- open all ports defined by Docker EXPOSE

#--- ISSUE:  uvicorn bug does not allow ctl-C break of fastapi through terminal
#--- WORKAROUND:  you have to run a docker or docker compose kill cmd;  eg docker kill <ctr_name>


#---    Docker build log
#       from python:3.10.9-slim-bullseye                    size:  4.21gb       time:  >yyys
