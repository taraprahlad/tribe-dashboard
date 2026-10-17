"""tribe brain response comparison (page of app.py)

pick a stimulus and two timesteps, see both brains side-by-side, and make a
difference map to see where the two brains diverge.

run the whole dashboard from the tribe-app folder with:
streamlit run app.py"""

import streamlit as st
import streamlit.components.v1 as components
from nilearn import plotting

from tribe_shared import load_metadata, load_preds, load_fsaverage, N_PER_HEMI

# the new idea this time is rendering the cached html data.
#
# this time we'll draw 3 brains per interaction, however the way a brain looks during a
# certain timestep doesn't change, so if we cache the html string keyed by
# (stimulus file, timestep), scrubbing back to a timestep we already viewed won't be laggy.
#
# cached functions take plain values (a file name string, int timesteps) and not arrays
# bc streamlit has to hash a function's arguments to determine whether it can reuse a
# cached result (strings and ints hash cleanly while arrays do not). therefore, a cached
# wrapper should take a file name + indices and look up the data itself :)

def make_html(data, title):  # this one isn't cached
    fsaverage = load_fsaverage()
    view = plotting.view_surf(
        surf_mesh=fsaverage["infl_left"],
        surf_map=data[:N_PER_HEMI],
        cmap="coolwarm",
        symmetric_cmap=True,
        title=title,
    )
    return view.get_iframe(width=500, height=400)

def demean(m):
    return m - m.mean()

@st.cache_data
def render_timestep(npy_file, t):
    preds = load_preds(npy_file)
    return make_html(preds[t], f"timestep {t}")

@st.cache_data
def render_difference(npy_file, a, b, mean_removed):
    preds = load_preds(npy_file)
    if mean_removed:
        diff = demean(preds[a]) - demean(preds[b])
    else:
        diff = preds[a] - preds[b]
    label = "mean-removed" if mean_removed else "raw"
    return make_html(diff, f"difference ({label}): timestep {a} - timestep {b}")

# this is where the page starts

st.title("tribe comparison view!")
st.write("Compare predicted brain activity at two different moments!")

meta = load_metadata()
title = st.selectbox("stimulus", meta["title"].tolist())
row = meta[meta["title"] == title].iloc[0]
npy_file = row["npy_file"]

preds = load_preds(npy_file)
n_timesteps = preds.shape[0]

col_a, col_b = st.columns(2)

with col_a:
    st.subheader("A")
    timestep_a = st.slider("timestep a!", 0, n_timesteps - 1, 0)
    components.html(render_timestep(npy_file, timestep_a), height=420)

with col_b:
    st.subheader("B")
    timestep_b = st.slider("timestep b!", 0, n_timesteps - 1, n_timesteps - 1)
    components.html(render_timestep(npy_file, timestep_b), height=420)

st.subheader("difference (A - B)")

mean_removed = st.checkbox("remove per-timestep mean", value=True)

if timestep_a == timestep_b:
    st.info("pick two different timesteps in order to see a difference map :)")
else:
    components.html(render_difference(npy_file, timestep_a, timestep_b, mean_removed), height=450)
    st.caption(
        "red = A is more active than B. blue = B is more active than A. "
        "white = the two moments agree/are the same"
    )