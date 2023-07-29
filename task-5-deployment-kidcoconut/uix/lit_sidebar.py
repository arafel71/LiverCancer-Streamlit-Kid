import streamlit as st
import importlib
from uix import lit_packages

from uix.pages import lit_home, lit_about, lit_diagnosis
from uix.pages import lit_qaConfigCheck

m_kblnTraceOn=False


#--- alt define sidebar pages
m_aryPages = {
    "Home":                         lit_home,           #--- TODO:  update
    "Diagnosis:  Single Tile":      lit_diagnosis,
    #"QA:  File Check":             lit_qaConfigCheck,         
    "About":                        lit_about
}


#--- define module-level vars
m_aryModNames = lit_packages.packages()
m_aryDescr = [] 
m_aryMods = []


def init():
    #--- upper panel
    with st.sidebar:
        kstrUrl_image = "bin/images/logo_omdena_saudi.png"
        st.sidebar.image(kstrUrl_image, width=200)

    #--- get radio selection
    strKey = st.sidebar.radio("rdoPageSel", list(m_aryPages.keys()), label_visibility="hidden")
    pagSel = m_aryPages[strKey]
    writePage(pagSel)


def writePage(uixFile):  
    #--- writes out the page for the selected combo

    # _reload_module(page)
    uixFile.run()

