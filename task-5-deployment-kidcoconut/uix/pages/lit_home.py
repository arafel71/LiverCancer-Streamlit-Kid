#--- about page
import streamlit as st

description = "Home"
def run():

    print("\nINFO (lit_home.run)  loading ", description, " page ...") 


    #st.markdown('### Home')
    #st.markdown('### Omdena Saudi Arabia')
    #st.markdown('### Detecting Liver Cancer from Histopathology WSI Using Deep Learning and Explainability')
    st.markdown('#### Background ')
    st.markdown('\
        Hepatocellular Carcinoma (HCC) is a primary liver malignancy, with \
        alarming global impact. It is the 4th most common cause of cancer \
        mortality worldwide, and the 6th most common malignancy overall. \
        \
        A patient\'s prognosis increases markedly with the speed of diagnosis \
        and treatment, however the rates of occurrence are increasing at an \
        alarming rate which will commensurately challenge the medical \
        community. \
        \
        There are already several tools and technologies available to assist \
        pathologists, however the current approach is ultimately constrained by \
        a number of factors including:  the rising demand, a limited supply \
        of skilled specialists, the time required to grow/replenish this talent \
        pool, and human factors which influence quality, accuracy, consistency, \
        and speed (timeliness). \
        ')

    st.markdown('#### Claim ')
    st.markdown('\
        It is the desire of this project team to increase the prognosis of \
        hepatocellular cancer patients.\
        \
        Machine Learning techniques, specifically Deep Learning and \
        Explainability (XAI) show promise in mimic\'ing the role of the \
        pathologist.  \
        \
        MLOps promises to establish a baseline for performance\
        and a basis for continuous process improvement.  This could greatly \
        reduce human factor elements while accelerating the times and \
        increasing the volumes of response.\
        \
        As a minimum, an ML application can serve as a supplement to the\
        pathologist, a teaching aide, a verification tool, or as a framework\
        for community collaboration and the advancement of quality diagnosis.\
    ')

    st.markdown('#### Objectives ')
    st.markdown('\
        A key objective of this project is to produce a deployed app that will\
        enable pathologists to upload a digital liver histopathology slide\
        image and then receive an output that classifies the segment as\
        malignant (or not). \
        \
        The utilization of Machine Learning and Explainability Techniques \
        to the traditional process of Liver Histopathology and HCC Diagnosis \
        could serve to greatly reduce the time to diagnosis and treatment. \
        \
    ')
    '''
    st.markdown(
        """

            Home page
        
        """,
            unsafe_allow_html=True,
        )
    <style>
        # MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
    </style>
    '''