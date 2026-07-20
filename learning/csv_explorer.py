"""a tiny csv explorer -- another throwaway streamlit app to learn file upload and dataframe display

run it with: streamlit run csv_explorer.py"""

import streamlit as st
import pandas as pd

st.title("csv explorer")
st.write("upload a csv file to explore it")

uploaded_file = st.file_uploader("choose a csv file", type =["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("your data")
    st.dataframe(df)

    st.subheader("summary")

    st.metric(label="rows:", value=len(df))
    st.metric(label="columns:", value=len(df.columns))

    column = st.selectbox("pick a column", df.columns)
    st.write(df[column].describe())

else:
    st.info("waiting for a csv file...")


