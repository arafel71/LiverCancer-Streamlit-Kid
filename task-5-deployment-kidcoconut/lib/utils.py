#--- notes:  
#   - this file is loaded by fastapi and streamlit, so keep it independant of those libs
#   - all path are relative to the appl working folder:  the parent of the lib folder;  ie ..\.. to this file

from pathlib import Path

pth_pwd = Path(__file__).resolve().parent               #--- should be \lib
pth_appRoot = pth_pwd.parent                            #--- ..

pth_root = str(pth_appRoot) + "/"

pth_bin = pth_root + "bin/"
pth_data = pth_root + "data/"
pth_lib = pth_root + "lib/"
pth_routes = pth_root + "routes/"
pth_templ = pth_root + "templ/"
pth_uix = pth_root + "uix/"

#--- bin paths
pth_binImages = pth_bin + "images/"
pth_binModels = pth_bin + "models/"

#--- data paths
pth_dtaApp = pth_data                                   #--- working folders for app data;  for docker, should be mapped to local host mount
pth_dtaDemoTiles = pth_data + "demo_tiles/"                   #--- dedicated area for demo data
pth_dtaTiles = pth_data + "tiles/"
pth_dtaWsi = pth_data + "wsi/"
pth_dtaTileSamples = pth_dtaDemoTiles + "raw/sample/"

#--- lib paths
pth_libModels = pth_lib + "models/"

#--- route paths
pth_rteApi = pth_routes + "api/"
pth_rteQa = pth_routes + "qa/"

m_klngMaxRecords = 100
m_klngSampleSize = 25
