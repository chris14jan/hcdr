import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#from ART_MCK.participant_plots import plot_score
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import numpy as np
from datetime import timedelta, date



def content():
    
    col1, col2 = st.beta_columns(2)
    
    col1.title('Dashboard')
    col1.write('Welcome to your dashboard')
    
    #Dashboard Filter: Choose Model to see best score:
    filter = col2.selectbox('Select a line to filter',["Model 1", "Model 2", "Model 3", "Ensemble"])
    st.markdown(f"{filter}")
    
