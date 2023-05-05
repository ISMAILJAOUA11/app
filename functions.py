import streamlit as st
import pandas as pd
import numpy as np
import io
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# 1)  Come scaricare la cartella creata su GitHub
#           git clone https://github.com/nemesiMark/app.git

# 2)  Come uplodare file su GitHub
#           git add .
#           git commit -m "nome modifica"
#           git push

# 3)  Come runnare su streamlit
#           steamlit run app0.py


def get_df_info(df):

    buffer = io.StringIO()
    df.info(buf=buffer)
    lines = buffer.getvalue().split('\n')

    # st.dataframe(lines[0:3])
    # st.dataframe(lines)

    for x in lines:
        st.text(x)

