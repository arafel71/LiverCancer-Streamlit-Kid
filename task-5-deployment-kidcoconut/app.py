'''
    toExecute:  (from root app folder) ... streamlit run app.py
'''
import streamlit as st
import uix.lit_sidebar as litSideBar


#--- streamlit:  specify title and logo
st.set_page_config(
            page_title='Omdena Saudi Arabia - Liver HCC Diagnosis with XAI', 
            #page_icon='https://cdn.freebiesupply.com/logos/thumbs/1x/nvidia-logo.png', 
            layout="wide")
st.header('\
    Detecting Liver Cancer from Histopathology WSI \
    using Deep Learning and Explainability (XAI)\
')
st.markdown('#### Dr. Shaista Hussain (Saudi Arabia Chapter Lead)')
st.markdown("##### Iain McKone (Deployment Lead) [LinkedIn](%s)" % "https://linkedin.com/in/iainmckone")
st.markdown('---')


#--- streamlit:  add a sidebar 
litSideBar.init()
