"""a tiny unit converter -- throwaway app for learning streamlit

run it w/ streamlit run learning/unit_converter.py"""

import streamlit as st

st.title("unit converter")
st.write("conversion between a few common units")

#letting the user pick whihc unit they want to convert to
conversion = st.selectbox(
    "what do you want to convert?",
    [
        "celsius to fahrenheit",
        "fahrenheit to celsius",
        "miles to kilometers",
        "kilometers to miles"
    ]
)
amount = st.number_input("enter a value:", value=0.0)
answer = None
if conversion == "celsius to fahrenheit":
    answer = amount * 9/5 + 32
elif conversion == "fahrenheit to celsius":
    answer = (amount - 32) * 5/9
elif conversion == "miles to kilometers":
    answer = amount * 1.60934
elif conversion == "kilometers to miles":
    answer = amount / 1.60934

if answer is not None:
    st.metric(label="answer", value=round(answer, 2))