from fastapi import APIRouter, Request, Response
from fastapi.responses import HTMLResponse
import numpy as np
import cv2
import os


import main as libMain
from lib import utils as libUtils


m_kstrFile = __file__
m_blnTraceOn = True

m_kstrPath_templ = libUtils.pth_templ


rteTiles = APIRouter()


#--- 
@rteTiles.get('/')
def api_tiles():
    return {
        "message": "tiles api endpoint - welcome to the endpoint for tile image processing"
    }


#--- 
@rteTiles.get('/raw/upload')
def api_tilesRawUpload():
    '''
        process an array of uploaded raw Tiles (from external app path)
        - cleanup all old raw images in /data/tiles/raw
        - save uploads to /data/tiles/raw
        - create tile class obj;  capture file path, size, zoomMagnif, etc
        - create array of tile class objs
        - return(s) json
            - ack tile/raw uploads with info/attribs
    '''
    return {
        "message": "tilesRawUpload endpoint - file processing of raw tile images"
    }


@rteTiles.get('/raw/norm')
def api_tilesRawNormalize(strPthTile):
    '''
        process an array of uploaded raw Tiles (from internal app path)
        - cleanup all old norm images in /data/tiles/norm
        - process tile normalization ops
        - save norm tiles to /data/tiles/norm
        - create tile class obj;  capture file path, size, zoomMagnif, etc
        - return(s) json
            - ack tile/norms with info/attribs
    '''
    #--- get file attributes
    strFilPath, strFilName = os.path.split(strPthTile)
    strPthRaw = strPthTile

    #--- load the tile as a binary object
    with open(strPthRaw,"rb") as filRaw: 
        imgRaw = filRaw.read()

    #--- Resize Tiles to 256x256
    #--- Note:  imgTile is a buffer object.  
    aryNp = np.frombuffer(imgRaw, np.uint8)
    imgTemp = cv2.imdecode(aryNp, cv2.IMREAD_COLOR)
    imgResized = cv2.resize(imgTemp, (256, 256))

    #--- save the normalized file 
    imgNorm = imgResized
    strPthNorm = "data/tiles/norm", strFilName
    with open(os.path.join(strPthNorm),"wb") as filNorm: 
        filNorm.write(imgResized.buffer)
    return strPthNorm
    """ return {
        "message": "tileRawNorm endpoint - normalization of raw tile images"
    }
 """

@rteTiles.get('/norm/upload')
def api_tilesNormUpload():
    '''
        process an array of uploaded norm Tiles (from external app path)
        - cleanup all old norm images in /data/tiles/norm
        - save uploads to /data/tiles/norm
        - create tile class obj;  capture file path, size, zoomMagnif, etc
        - create array of tile class objs
        - return(s) json
            - ack tile/norm uploads with info/attribs
    '''
    return {
        "message": "tilesNormUpload endpoint - file processing of norm tile images"
    }


@rteTiles.get('/norm/preprocess')
def api_tilesNormPreprocess():
    '''
        preprocess an array of uploaded norm Tiles (from internal app path)
        - perform remaining pre-processing of tiles prior to model prediction
        - cleanup all old preproc images in /data/tiles/preproc
        - save preproc tiles to /data/tiles/preproc
        - create tile class obj;  capture file path, size, zoomMagnif, etc
        - return(s) json
            - ack tile/preproc with info/attribs
    '''
    return {
        "message": "tileNormPreprocess endpoint - preprocessing of normalized tile images"
    }


@rteTiles.get('/preproc/upload')
def api_tilesPreprocUpload():
    '''
        process an array of uploaded preprocessed Tiles (from external app path)
        - cleanup all old preproc images in /data/tiles/preproc
        - save uploads to /data/tiles/preproc
        - create tile class obj;  capture file path, size, zoomMagnif, etc
        - create array of tile class objs
        - return(s) json
            - ack tile/preproc uploads with info/attribs
    '''
    return {
        "message": "tilesPreprocUpload endpoint - manage upload of preprocessed tile images, in prep for modelling/prdictions"
    }


@rteTiles.get('/preproc/augment')
def api_tilesPreprocAugment():
    '''
        process an array of uploaded preprocessed tiles (from internal app path)
        - cleanup all old augmented tiles in /data/tiles/augm
        - perform augments of tiles prior to model prediction (translation, rotation, transforms)
        - save augmented tiles to /data/tiles/augm
        - create tile class obj;  capture file path, size, zoomMagnif, etc
        - return(s) json
            - ack tile/augm with info/attribs
    '''
    return {
        "message": "tilePreprocAugment endpoint - augment tile images"
    }


@rteTiles.get('/augm/upload')
def api_tilesAugmUpload():
    '''
        process an array of augmented tiles (from external app path)
        - cleanup all old augm images in /data/tiles/augm
        - save uploads to /data/tiles/augm
        - create tile class obj;  capture file path, size, zoomMagnif, etc
        - create array of tile class objs
        - return(s) json
            - ack tile/augm uploads with info/attribs
    '''
    return {
        "message": "tilesAugmUpload endpoint - manage upload of augmented tile images, in prep for modelling/predictions"
    }


#--- 
@rteTiles.get('/raw/predict')
def api_tileRawPredict():
    return {
        "message": "tile_rawPredict api endpoint - welcome to the endpoint for tile predictions"
    }


#--- 
@rteTiles.get('/norm/segment')
def api_tileNormPredict():
    return {
        "message": "tile_normPredict api endpoint - welcome to the endpoint for tile predictions"
    }

#--- 
@rteTiles.get('/norm/predict')
def api_tileNormPredict():
    return {
        "message": "tile_normPredict api endpoint - welcome to the endpoint for tile predictions"
    }


#--- 
@rteTiles.get('/preproc/predict')
def api_tilePreprocPredict():
    return {
        "message": "tile_preprocPredict api endpoint - welcome to the endpoint for tile predictions"
    }


#--- 
@rteTiles.get('/augm/predict')
def api_tileAugmPredict():
    return {
        "message": "tile_augmPredict api endpoint - welcome to the endpoint for tile predictions"
    }