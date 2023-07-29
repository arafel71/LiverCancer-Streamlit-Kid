'''
    purpose:     fastAPI routing 
'''

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi import APIRouter, Request, Response
from fastapi.templating import Jinja2Templates
import uvicorn

#--- import custom libraries
import lib.utils as libUtils


#--- imported route handlers
from routes.api.rte_api import rteApi
from routes.api.rte_wsi import rteWsi
from routes.api.rte_tiles import rteTiles


#--- fastAPI self doc descriptors
description = """
    Omdena Saudi Arabia:  Liver Cancer HCC Diagnosis with XAI
    
    <insert purpose>

    ## key business benefit #1
    ## key business benefit #2
    ## key business benefit #3

    You will be able to:
    * key feature #1
    * key feature #2
    * key feature #3
"""

app = FastAPI(
    title="App:  Omdena Saudi Arabia - Liver Cancer HCC Diagnosis with XAI",
    description=description,
    version="0.0.1",
    terms_of_service="http://example.com/terms/",
    contact={
        "name": "Iain McKone",
        "email": "iain.mckone@gmail.com",
    },
    license_info={
        "name": "Apache 2.0",
        "url": "https://www.apache.org/licenses/LICENSE-2.0.html",
    },
)


#--- configure route handlers
app.include_router(rteWsi, prefix="/api/wsi")
app.include_router(rteTiles, prefix="/api/tiles")
app.include_router(rteApi, prefix="/api")

#app.include_router(rteQa, prefix="/qa")


m_kstrPath_templ = libUtils.pth_templ
m_templRef = Jinja2Templates(directory=str(m_kstrPath_templ))


def get_jinja2Templ(request: Request, pdfResults, strParamTitle, lngNumRecords, blnIsTrain=False, blnIsSample=False):
    lngNumRecords = min(lngNumRecords, libUtils.m_klngMaxRecords)
    if (blnIsTrain):  strParamTitle = strParamTitle + " - Training Data"
    if (not blnIsTrain):  strParamTitle = strParamTitle + " - Test Data"
    if (blnIsSample):  lngNumRecords = libUtils.m_klngSampleSize
    strParamTitle = strParamTitle + " - max " + str(lngNumRecords) + " rows"

    kstrTempl = 'templ_showDataframe.html'
    jsonContext = {'request': request, 
                'paramTitle': strParamTitle,
                'paramDataframe': pdfResults.sample(lngNumRecords).to_html(classes='table table-striped')
            }
    result = m_templRef.TemplateResponse(kstrTempl, jsonContext)
    return result


#--- get main ui/ux entry point
@app.get('/')
def index():
    return {
        "message": "Landing page:  Omdena Saudi Arabia - Liver HCC Diagnosis with XAI"
    }



if __name__ == '__main__':
    uvicorn.run("main:app", host="0.0.0.0", port=49300, reload=True)
#CMD ["uvicorn", "main:app", "--host=0.0.0.0", "--reload"]
