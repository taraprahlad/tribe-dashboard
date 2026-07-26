"""tribe stimulus library browser: local

this should let you browse through a selection of pre-computed predictions. you can pick a stimulus from
the dropdown, scrub through its timesteps and see the predicted brain response. 

run it from inside the tribe app folder with the following: streamlit run library_app.py

also, note that it will expect library metadata and the preds.npy file."""

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from nilearn import datasets, plotting

st.set_page_config(layout = "wide")

n_per_hemi = 10242
LIBRARY_DIR = "library"

# cached loaders!

@st.cache_data
def load_metadata():
    return pd.read_csv(f"{LIBRARY_DIR}/library_metadata.csv")

@st.cache_data
def load_preds(npy_file):
    """Load one stimulus's prediction array. Cached per file."""
    return np.load(f"{LIBRARY_DIR}/{npy_file}")
 
 
@st.cache_data
def load_fsaverage():
    return datasets.fetch_surf_fsaverage(mesh="fsaverage5")

# rendering!

import matplotlib.pyplot as plt

@st.cache_data
def render(npy_file, timestep):
    preds = load_preds(npy_file)
    fsaverage = load_fsaverage()
    fig = plotting.plot_surf_stat_map(
        surf_mesh = fsaverage["infl_left"],
        stat_map = preds[timestep][:n_per_hemi],
        cmap = "coolwarm",
        symmetric_cbar = True,
        title = f"timestep {timestep}",
    )
    return fig

# the actual page

st.title("tribe stimulus library!")
st.write("you can browse predicted brain responses across stimuli here.")

meta = load_metadata()

title = st.selectbox("choose a stimulus", meta["title"])
row = meta[meta["title"] == title].iloc[0]
npy_file = row["npy_file"]
n_timesteps = int(row["n_timesteps"])

st.caption(row["notes"])

timestep = st.slider("timestep", 0, n_timesteps - 1, 0)

st.pyplot(render(npy_file, timestep))
st.caption(
    f"{title} -- timestep {timestep} # {n_timesteps - 1}."
    "red = above baseline, blue = below baseline, white = near baseline."
)