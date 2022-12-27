import streamlit as st 
import  matplotlib.pyplot as plt
import  pandas as pd

import time
import joblib
import shap
import plotly.graph_objects as go


html_txt = """
    <div style="background-color: Cyan; padding:5px; border-radius:8px">
    <h1 style="color: black; text-align:center">HOME CREDIT DEFAULT RISK</h1>
    </div>

    <div>  
    </div>

    <div style="background-color: white; padding:5px; border-radius:8px">
        Nous Construisons un modèle de scoring qui donnera une prédiction sur la probabilité de faillite d'un client de façon automatique.

    </div>

    <div style="background-color: white; padding:5px; border-radius:8px">
        Construire un dashboard interactif à destination des gestionnaires de la relation client permettant d'interpréter les prédictions faites par le modèle, et d’améliorer la 
        connaissance client des chargés de relation client.

    </div>




    """
st.markdown(html_txt, unsafe_allow_html=True)

