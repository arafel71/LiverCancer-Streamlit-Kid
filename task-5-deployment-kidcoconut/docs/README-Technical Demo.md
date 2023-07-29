# Omdena Saudi - Liver HCC Diagnosis with ML and XAI


## Table of Contents
1.  Background
2.  Overview
3.  How to run the demo
4.  How to deploy
5.  How to build
6.  Roadmap




## Section 1:  Background
xxx



## Section 2:  Overview
yyyy

### 2.1     Technical Components/Requirements
- dev environment
  - gitHub: 
- release environment
- user environment


### 2.2     Technical Limitations
- Omdena GitHub/GitLFS
- Huggingface



## Section 3:  How to run the demo
### 3.1     Huggingface Spaces:
- Streamlit:  https://huggingface.co/spaces/kidcoconut/spcstm_omdenasaudi_liverhccxai
- Docker:     TBD


### 3.2     Local env:      
- Streamlit:    http://localhost:39400
- FastAPI:      http://localhost:39500


### 3.3     Docker:      
- Streamlit:    http://localhost:49400
- FastAPI:      http://localhost:49500




## Section 4:  How to deploy
### 4.1     Huggingface Spaces:
- Streamlit:  TBD
- Docker:     TBD
- Data:       TBD
- Model:      TBD


### 4.2     Local env:      (from task-5-deployment folder)
- streamlit:    streamlit run lit_index.py --server.port 48400
- fastAPI:      uvicorn main:app --reload --workers 1 --host 0.0.0.0 --port 48300
- docker:       
- dockerHub:



## Section 5:  How to build
### 5.1     Huggingface Spaces:


### 5.2     Local Dev Environment
- conda:        
- pip:          
- git:          


### 5.3     Docker Image
- build:        docker build -t img_apiclaimanoms:demo .
- run:          docker run -p 48300:8000 --name ctr_apiClaimAnoms img_apiclaimanoms:demo



## Section 6:  Roadmap
- model:        
- streamlit:    
- fastapi:      
- model serve:  