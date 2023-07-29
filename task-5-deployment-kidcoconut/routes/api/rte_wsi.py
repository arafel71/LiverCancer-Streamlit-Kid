from fastapi import APIRouter, Request, Response
from fastapi.responses import HTMLResponse


import main as libMain
from lib import utils as libUtils


m_kstrFile = __file__
m_blnTraceOn = True

m_kstrPath_templ = libUtils.pth_templ


rteWsi = APIRouter()


#--- 
@rteWsi.get('/')
def api_wsi():
    return {
        "message": "wsi api endpoint - welcome to the endpoint for wsi image processing"
    }


#--- 
@rteWsi.get('/upload')
def api_wsiUpload():
    '''
        process a single uploaded WSI image (from external app path)
        - cleanup all old WSI images in /data/wsi/raw
        - save upload to /data/wsi/raw
        - create wsi class obj;  capture file path, size, zoomMagnif, etc
        - return(s) json
            - ack wsi upload with info/attribs
    '''
    return {
        "message": "wsiUpload endpoint - file processing of one uploaded wsi image"
    }


#--- 
@rteWsi.get('/chunk')
def api_wsiChunk():
    '''
        process a single WSI image  (from internal app path)
        - create wsi class obj;  capture file path, size, zoomMagnif, etc
        - kick off tile chunking process;  
        - save tiles to /data/tiles/raw
        - return(s) json
            - ack wsi upload with info/attribs
            - ack of tiles created:  total count;  names, paths, attribs (dimensions)
    '''
    return {
        "message": "wsiLoad endpoint - for chunking of wsi image to one or more tiles"
    }
