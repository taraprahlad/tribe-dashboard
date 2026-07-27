"""tribe region explorer! (local)

we're getting to the good stuff hehe. this should let you pick a brain region, see
how active it is over the course of a stimulus, and where it sits on the brain. it uses
the destrieux atlas to map the model's 20484 anonymous vertices to the name
anatomical regions.

i'm tired of telling you how to run it, i think you get the gist by now."""

import faulthandler
faulthandler.enable()

import numpy as np
import pandas as pd
pd.set_option("future.infer_string", False)
import matplotlib.pyplot as plt
import streamlit as st
from nilearn import datasets, plotting

st.set_page_config(layout = "wide")

n_per_hemi = 10242
LIBRARY_DIR = "library"

# caching

@st.cache_data
def load_metadata():
    return pd.read_csv(f"{LIBRARY_DIR}/library_metadata.csv")
 
 
@st.cache_data
def load_preds(npy_file):
    return np.load(f"{LIBRARY_DIR}/{npy_file}")
 
 
@st.cache_data
def load_fsaverage():
    return datasets.fetch_surf_fsaverage(mesh="fsaverage5")

@st.cache_data
def load_atlas(): # should return left hemi -> region index array and list of region names
    region_of_vertex = np.load(f"{LIBRARY_DIR}/atlas_map_left.npy")
    with open(f"{LIBRARY_DIR}/atlas_labels.txt") as f:
        labels = [str(line) for line in f.read().splitlines()]
    return region_of_vertex, labels

# for a region, average the prediction over its vertices
def region_timeseries(preds, region_of_vertex, region_index):
    left = preds[:, :n_per_hemi]
    mask = (region_of_vertex == region_index)
    mean = left[:, mask].mean(axis = 1)
    return mean

# page

st.title("tribe region explorer!")

meta = load_metadata()
region_of_vertex, labels = load_atlas()

title = st.selectbox("stimulus", meta["title"])
row = meta[meta["title"] == title].iloc[0]
preds = load_preds(row["npy_file"])

region_names = [str(name) for i, name in enumerate(labels) if i != 0]
region_name = st.selectbox("brain region", region_names)
region_index = labels.index(region_name)

ts = region_timeseries(preds, region_of_vertex, region_index)

st.subheader(f"{region_name} -- activation over time")
fig, ax = plt.subplots(figsize = (8, 3))
ax.plot(ts)
ax.set_xlabel("timestep")
ax.set_ylabel("mean activation")
ax.axhline(0, color = "gray", linewidth = 0.5)
st.pyplot(fig)
plt.close(fig)

st.caption(
    f"mean predicted activation of {region_name} across '{title}'."
    "above the gray line = more active than baseline"
)
