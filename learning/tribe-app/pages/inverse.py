"""the inverse of the region selector app: local

this lets you pick a timestamp and get a ranked list of the most to least active brain regions."""

import numpy as np
import pandas as pd
import streamlit as st

pd.set_option("future.infer_string", False)

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
def load_atlas():
    region_of_vertex = np.load(f"{LIBRARY_DIR}/atlas_map_left.npy")
    with open(f"{LIBRARY_DIR}/atlas_labels.txt") as f:
        labels = [str(line) for line in f.read().splitlines()]
    return region_of_vertex, labels

# ranked region list
def rank_regions_at_timestep(preds, region_of_vertex, labels, timestep):
    left = preds[timestep, :n_per_hemi]
    rows = []
    for region_index, name in enumerate(labels):
        if region_index == 0:
            continue
        mask = (region_of_vertex == region_index)
        if not mask.any():
            continue
        region_mean = left[mask].mean()
        rows.append({"region": name, "mean_activation": region_mean})

    df = pd.DataFrame(rows)
    return df.sort_values("mean_activation", ascending = False).reset_index(drop = True)

# page
st.title("tribe inverse view")
st.write("which brain regions are most active at a given moment?")

meta = load_metadata()
region_of_vertex, labels = load_atlas()

# pick stimulus
title = st.selectbox("stimulus", meta["title"])
row = meta[meta["title"] == title].iloc[0]
preds = load_preds(row["npy_file"])
n_timesteps = int(row["n_timesteps"])

timestep = st.slider("timestep", 0, n_timesteps - 1, 0)

ranked = rank_regions_at_timestep(preds, region_of_vertex, labels, timestep)
st.subheader(f"top regions at timestep {timestep} of '{title}'")

top10 = ranked.head(10)

md = "| Rank | Region | Mean activation |\n|---|---|---|\n"
for i, (_, r) in enumerate(top10.iterrows(), start=1):
    md += f"| {i} | {r['region']} | {r['mean_activation']:.4f} |\n"

st.markdown(md)
