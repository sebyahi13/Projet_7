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
    """
st.markdown(html_txt, unsafe_allow_html=True)
