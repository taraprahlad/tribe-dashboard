"""tribe_shared.py: paths and cached loaders shared by every page of the app."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from nilearn import datasets
from nilearn.datasets import load_fsaverage as nilearn_load_fsaverage   # renamed: avoids the week 11 name clash

ROOT = Path(__file__).resolve().parent        # tribe-app/
LIBRARY = ROOT / "library"
ATTRIBUTION = LIBRARY / "attribution"
N_PER_HEMI = 10242                              # fsaverage5 vertices per hemisphere


@st.cache_data
def load_metadata():
    """The stimulus library table. fillna('') so blank cells aren't float NaN."""
    return pd.read_csv(LIBRARY / "library_metadata.csv").fillna("")


@st.cache_data
def load_preds(npy_file):
    """One stimulus's predictions, shape (n_timesteps, 20484).
    Uses only the file name, so rows stored as 'library/x.npy' or 'x.npy' both work."""
    return np.load(LIBRARY / Path(npy_file).name)


@st.cache_data
def load_fsaverage():
    """fsaverage5 surface file paths (used by the library and compare pages)."""
    return datasets.fetch_surf_fsaverage(mesh="fsaverage5")


@st.cache_data
def load_left_mesh():
    """Left inflated fsaverage5 mesh as (coords, faces), guaranteed 10,242 vertices.
    (fetch_surf_fsaverage's infl_left loaded at 163,842 vertices in week 11.)"""
    left = nilearn_load_fsaverage("fsaverage5")["inflated"].parts["left"]
    coords, faces = np.asarray(left.coordinates), np.asarray(left.faces)
    assert coords.shape[0] == N_PER_HEMI, f"mesh has {coords.shape[0]} vertices"
    return coords, faces


@st.cache_data
def load_atlas():
    """Precomputed Destrieux atlas (left hemisphere): region index per vertex + label names."""
    region_of_vertex = np.load(LIBRARY / "atlas_map_left.npy")
    labels = (LIBRARY / "atlas_labels.txt").read_text().splitlines()
    return region_of_vertex, labels


@st.cache_data
def load_attribution_json(name):
    with open(ATTRIBUTION / name) as f:
        return json.load(f)


@st.cache_data
def load_attribution_npy(name):
    return np.load(ATTRIBUTION / name)