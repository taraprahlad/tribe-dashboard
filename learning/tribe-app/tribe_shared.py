"""tribe_shared.py: paths and cached loaders shared by every page of the app."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from nilearn import datasets

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
    return datasets.fetch_surf_fsaverage(mesh="fsaverage5")


@st.cache_data
def load_destrieux():
    return datasets.fetch_atlas_surf_destrieux()


@st.cache_data
def load_attribution_json(name):
    with open(ATTRIBUTION / name) as f:
        return json.load(f)


@st.cache_data
def load_attribution_npy(name):
    return np.load(ATTRIBUTION / name)