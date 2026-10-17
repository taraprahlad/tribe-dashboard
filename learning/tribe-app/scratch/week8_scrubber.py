"""tribe brain response dashboard (local edition bc google colab is being a bitch)

interactive app -- you should be able to pick a timestamp with the slider and see the predicted brain activity for that
moment selected. runs entirely locally on your mac against preds.npy (a saved prediction).

run it with: streamlit run app.py

note this is for me, not you the user.

new idea: caching!

since streamlit runs this whole thing from the top to bottom every time you interact with it, we're going
to use smth called caching -- some things are expensive and are only going to be run once (i.e. loading the 
prediction file and downloading the fsaverage brain mesh)

@st.cache_data tells streamlit "yo run the function once and then remember the result, and on future re-runs
give us the saved result instead of re-running the entire thing. we're doing this so the slider isn't laggy."""

import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from nilearn import datasets, plotting

@st.cache_data
def load_preds():
    return np.load("preds.npy")

@st.cache_data
def load_fsaverage():
    return datasets.fetch_surf_fsaverage(mesh = "fsaverage5")


st.title("tribe brain response explorer")
st.write(
    "predicted brain activity from a stimulus, moment by moment!"
    "use the slider to scrub through time :)"
)


preds = load_preds()
fsaverage = load_fsaverage()

n_timesteps = preds.shape[0]

timestep = st.slider(
    "timestep",
    min_value = 0,
    max_value = n_timesteps - 1,
    value = 0
)

one_timestep = preds[timestep]
n_per_hemi = 10242
left_data = one_timestep[:n_per_hemi]
right_data = one_timestep[n_per_hemi:]


view = plotting.view_surf(
    surf_mesh = fsaverage["infl_left"],
    surf_map = left_data,
    cmap = "coolwarm",
    symmetric_cmap = True,
    title = f"left hemisphere -- timestep {timestep}",
)

view.save_as_html("_brain.html")
with open("_brain.html") as f:
    brain_html = f.read()

components.html(brain_html, height = 500)

st.caption(
    f"showing timestep {timestep} of {n_timesteps - 1}."
    "red = above baseline, blue = below baseline, white = near baseline"
)
