"""the inverse of the region selector (page of app.py)

this lets you pick a timestep and get a ranked list of the most to least active
brain regions (left hemisphere, destrieux atlas)."""

import pandas as pd
import streamlit as st

from tribe_shared import load_metadata, load_preds, load_atlas, N_PER_HEMI

# ranked region list
def rank_regions_at_timestep(preds, region_of_vertex, labels, timestep):
    left = preds[timestep, :N_PER_HEMI]
    rows = []
    for region_index, name in enumerate(labels):
        if region_index == 0:          # 0 = unknown / medial wall
            continue
        mask = (region_of_vertex == region_index)
        if not mask.any():
            continue
        rows.append({"region": name, "mean_activation": left[mask].mean()})

    df = pd.DataFrame(rows)
    return df.sort_values("mean_activation", ascending=False).reset_index(drop=True)

# page
st.title("tribe inverse view!")
st.write("which brain regions are most active at a given moment?")

meta = load_metadata()
region_of_vertex, labels = load_atlas()

# pick stimulus
title = st.selectbox("stimulus", meta["title"].tolist())
row = meta[meta["title"] == title].iloc[0]
preds = load_preds(row["npy_file"])
n_timesteps = preds.shape[0]

timestep = st.slider("timestep", 0, n_timesteps - 1, 0)

ranked = rank_regions_at_timestep(preds, region_of_vertex, labels, timestep)
st.subheader(f"top regions at timestep {timestep} of '{title}'")

top10 = ranked.head(10)

md = "| rank | region | mean activation |\n|---|---|---|\n"
for i, (_, r) in enumerate(top10.iterrows(), start=1):
    md += f"| {i} | {r['region']} | {r['mean_activation']:.4f} |\n"

st.markdown(md)