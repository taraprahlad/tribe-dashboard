"""app.py: entry point for the TRIBE v2 interpretability dashboard.
run with:  streamlit run app.py
"""
import faulthandler; faulthandler.enable()
import matplotlib
matplotlib.use("Agg")                          # macOS: avoids the segfault; must precede pyplot
import pandas as pd
pd.set_option("future.infer_string", False)    # the week 12 pyarrow selectbox segfault
import streamlit as st

st.set_page_config(page_title="TRIBE v2 Interpretability", layout="wide")

pg = st.navigation([
    st.Page("pages/library.py", title="Library"),
    st.Page("pages/compare.py", title="Compare timesteps"),
    st.Page("pages/regions.py", title="Regions"),
    st.Page("pages/inverse.py", title="Most active regions"),
    st.Page("pages/attribution.py", title="Attribution"),   # add once the page exists
])
pg.run()