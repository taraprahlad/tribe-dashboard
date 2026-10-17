"""tribe stimulus library browser (page of app.py)

this should let you browse through a selection of pre-computed predictions. you can pick a
stimulus from the dropdown, scrub through its timesteps and see the predicted brain response
(left hemisphere).

run the whole dashboard from the tribe-app folder with: streamlit run app.py"""

import io

import matplotlib.pyplot as plt
import streamlit as st
from nilearn import plotting

from tribe_shared import load_metadata, load_preds, load_fsaverage, N_PER_HEMI

# rendering!
# the figure is drawn, saved as png bytes, and closed right away. caching the bytes
# (instead of the figure) means no figures pile up in memory: that pile-up was
# the week 12 segfault.

@st.cache_data
def render_png(npy_file, timestep):
    preds = load_preds(npy_file)
    fsaverage = load_fsaverage()
    fig = plotting.plot_surf_stat_map(
        surf_mesh=fsaverage["infl_left"],
        stat_map=preds[timestep][:N_PER_HEMI],
        cmap="coolwarm",
        symmetric_cbar=True,
        title=f"timestep {timestep}",
    )
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    plt.close(fig)
    return buf.getvalue()

# the actual page

st.title("tribe stimulus library!")
st.write("you can browse predicted brain responses across stimuli here.")

meta = load_metadata()

title = st.selectbox("choose a stimulus", meta["title"].tolist())
row = meta[meta["title"] == title].iloc[0]
npy_file = row["npy_file"]
n_timesteps = load_preds(npy_file).shape[0]

st.caption(row["notes"])
if row["credit"]:
    st.caption(f"source: {row['credit']}")

timestep = st.slider("timestep", 0, n_timesteps - 1, 0)

st.image(render_png(npy_file, timestep))
st.caption(
    f"{title}: timestep {timestep} of {n_timesteps - 1}. "
    "red = above baseline, blue = below baseline, white = near baseline."
)