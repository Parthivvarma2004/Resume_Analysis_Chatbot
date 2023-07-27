from collections import namedtuple
import altair as alt
import math
import pandas as pd
import streamlit as st
import numpy as np

"""
# Welcome to our chatbot website!

Feel free to upload resume files below to analyze them by talking to our chatbot

-- Parthiv and Sadman
"""

uploaded_files = st.file_uploader("Choose a PDF file", accept_multiple_files=True)
for uploaded_file in uploaded_files:
    bytes_data = uploaded_file.read()
    st.write("filename:", uploaded_file.name)
    st.write(bytes_data)

#If you have any questions, checkout our [documentation](add a link to our instruction manual here) 
