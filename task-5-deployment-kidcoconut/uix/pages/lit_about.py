#--- about page
import streamlit as st

description = "About"
def run():

    print("\nINFO (lit_about.run)  loading ", description, " page ...") 

    #--- 
    #st.experimental_memo.clear()            #--- try to clear cache each time this page is hit
    #st.cache_data.clear()

    #st.markdown('### About')
    #st.markdown('### Omdena Saudi Arabia')
    #st.markdown('### Detecting Liver Cancer from Histopathology WSI Using Deep Learning and Explainability')
    #st.markdown('#### Dr. Shaista Hussain (Saudi Arabia Chapter Lead)')
    #st.markdown('##### Deployment Lead:  Iain McKone')
    st.markdown('##### Project Url:  https://github.com/OmdenaAI/saudi-arabia-histopathology-detection')
    '''
    st.markdown(
        """
            About page
        """,
            unsafe_allow_html=True,
        )
    '''