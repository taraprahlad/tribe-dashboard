"""tribe brain response comparison (local)

this will let you pick two timesteps, see both brains. side-by-side and make a difference map
to see where the two brains diverge.

you can run it from the tribe-app folder with the following:
streamlit run compare_app.py"""

import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from nilearn import datasets, plotting

n_per_hemi = 10242

@st.cache_data
def load_preds():
    return np.load("preds.npy")

@st.cache_data
def load_fsaverage():
    return datasets.fetch_surf_fsaverage(mesh = "fsaverage5")

# the new idea this time is rendering the cached html data.
#
# this time we'll draw 3 brains per interaction, however the way a brain looks during a
# certain timestep doesn't change, so if we cache the html string keyed by timestep number
# scrubbing back to a timestep we already viewed won't be laggy.
#
# cached functions take plain ints and not arrays bc streamlit has to
# hash a function's arguments to determine whether it can reuse a cached result (ints
# hash cleanly while arrays do not). therefore, a cached wrapper should take indices
# and look up data themselves :)

def make_html(data, title): #this one isn't cached
    fsaverage = load_fsaverage()
    view = plotting.view_surf(
        surf_mesh = fsaverage["infl_left"],
        surf_map = data[:n_per_hemi],
        cmap = "coolwarm",
        symmetric_cmap = True,
        title = title,
    )
    return view.get_iframe(width = 500, height = 400)

def demean(m):
    return m - m.mean()

@st.cache_data
def render_timestep(t):
    preds = load_preds()
    return make_html(preds[t], f"timestep {t}")

@st.cache_data
def render_difference(a, b, mean_removed):
    preds = load_preds()
    if mean_removed:
        diff = demean(preds[a]) - demean(preds[b])
    else:
        diff = preds[a] - preds[b]
    label = "mean-removed" if mean_removed else "raw"
    return make_html(diff, f"difference ({label}): timestep {a} - timestep {b}")

#this is where the page starts

st.set_page_config(layout = "wide")
st.title("TRIBE Comparison View")
st.write("Compare predicted brain activity two different moments!")

preds = load_preds()
n_timesteps = preds.shape[0]

col_a, col_b = st.columns(2)

with col_a:
    st.subheader("A")
    timestep_a = st.slider("timestep a!", 0, n_timesteps - 1, 0)
    components.html(render_timestep(timestep_a), height = 420)

with col_b:
    st.subheader("B")
    timestep_b = st.slider("timestep b!", 0, n_timesteps - 1, n_timesteps - 1)
    components.html(render_timestep(timestep_b), height = 420)

st.subheader("difference (A - B)")

mean_removed = st.checkbox("remove per-timestep mean", value = True)
html = render_difference(timestep_a, timestep_b, mean_removed)

if timestep_a == timestep_b:
    st.info("pick two different timesteps in order to see a difference map :)")
else:
    components.html(render_difference(timestep_a, timestep_b, mean_removed), height = 450)
    st.caption(
        "red = A is more active than B. blue = B is more active than A." \
        "white = the two moments agree/are the same"
    )