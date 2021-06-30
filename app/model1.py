import streamlit as st
import pandas as pd
import numpy as np

def content(participant_id):
    col1, col2 = st.beta_columns(2)
    col1.title("General Recommendations")
    col1.write("Here you can see the general affect of behaviour on you productivity in the workplace")
    
    col2.selectbox('Select a line to filter',["NN with 2 Layers", "NN with 4 Layers"])
    
    st.markdown("A NN with more layers seems to perform better, all else equal (including total number of neurons).")