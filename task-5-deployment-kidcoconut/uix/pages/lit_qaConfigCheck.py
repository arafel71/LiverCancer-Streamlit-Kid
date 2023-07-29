#--- about page
import streamlit as st
import sys, os
import pandas as pd

import lib.utils as libUtils


description = "QA:  Config Check"
def run():

    print("\nINFO (lit_config.run)  loading ", description, " page ...") 

    #--- 
    #st.experimental_memo.clear()            #--- try to clear cache each time this page is hit
    #st.cache_data.clear()

    st.markdown('### Configuration Check')

    #--- check that base folders exist
    #--- list raw WSIs
    lstWSI = os.listdir(libUtils.pth_dtaWsi + "raw/")
    print("TRACE: ", lstWSI)
    st.dataframe(
        pd.DataFrame({"Raw WSI":  lstWSI,}),
        use_container_width=True
    )

    #--- list raw Tiles
    lstTiles = os.listdir(libUtils.pth_dtaTiles + "raw/")
    print("TRACE: ", lstTiles)
    st.dataframe(
        pd.DataFrame({"Raw Tiles":  lstTiles,}),
        use_container_width=True
    )

    #--- list raw demo Tiles
    lstDemo = os.listdir(libUtils.pth_dtaDemoTiles + "raw/")
    print("TRACE: ", lstDemo)
    st.dataframe(
        pd.DataFrame({"Raw Demo Tiles":  lstDemo,}),
        use_container_width=True
    )


    st.markdown('''
        <style>
            [data-testid="stMarkdownContainer"] ul{
                list-style-position: inside;
            }
        </style>
        ''', unsafe_allow_html=True)
    

#    st.markdown(
        # st.footer(
        # """
        #     Configuration Check page
        # """,
        #     unsafe_allow_html=True,
        # )
    
    cssFooter="""
        <style>
            a:link, 
            a:visited{
                color: blue;
                background-color: transparent;
                text-decoration: underline;
            }
            a:hover,  a:active {
                color: red;
                background-color: transparent;
                text-decoration: underline;
                }
            .footer {
                position: fixed;
                left: 0; bottom: 0; width: 100%;
                background-color: white;
                color: black;
                text-align: center;
            }
        </style>
        <div class="footer">
            <p>Configuration Check Page</p>
        </div>
    """
    st.markdown(cssFooter, unsafe_allow_html=True)