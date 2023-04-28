import streamlit as st
import pandas as pd

uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is not None:
    st.write("yay, you are not an idiot")