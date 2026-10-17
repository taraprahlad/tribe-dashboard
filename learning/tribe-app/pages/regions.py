"""tribe region explorer! (page of app.py)

we're getting to the good stuff hehe. this should let you pick a brain region, see
how active it is over the course of a stimulus, and where it sits on the brain. it uses
the destrieux atlas to map the model's 20484 anonymous vertices to named anatomical
regions (left hemisphere).

i'm tired of telling you how to run it, i think you get the gist by now."""

import io

import matplotlib.pyplot as plt
import streamlit as st
from nilearn import plotting

from tribe_shared import load_metadata, load_preds, load_atlas, load_left_mesh, N_PER_HEMI

# for a region, average the prediction over its vertices
def region_timeseries(preds, region_of_vertex, region_index):
    left = preds[:, :N_PER_HEMI]
    mask = (region_of_vertex == region_index)
    return left[:, mask].mean(axis=1)

# where the region sits on the brain.
# takes only the region index (an int) so it caches cleanly, and looks up the
# mesh + atlas itself. the figure becomes png bytes and is closed right away,
# same as the library page, so figures don't pile up in memory.
@st.cache_data
def render_region_png(region_index):
    coords, faces = load_left_mesh()             # verified 10,242-vertex fsaverage5 mesh
    region_of_vertex, _ = load_atlas()
    highlight = (region_of_vertex == region_index).astype(float)
    fig = plotting.plot_surf_stat_map(
        surf_mesh=[coords, faces],
        stat_map=highlight,
        hemi="left",
        view="lateral",
        cmap="autumn",
        threshold=0.5,        # everything outside the region stays grey
        colorbar=False,
    )
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    plt.close(fig)
    return buf.getvalue()

# page

st.title("tribe region explorer!")

meta = load_metadata()
region_of_vertex, labels = load_atlas()

title = st.selectbox("stimulus", meta["title"].tolist())
row = meta[meta["title"] == title].iloc[0]
preds = load_preds(row["npy_file"])

# skip index 0 (unknown / medial wall) and any label with no vertices on this mesh
region_names = [name for i, name in enumerate(labels)
                if i != 0 and (region_of_vertex == i).any()]
region_name = st.selectbox("brain region", region_names)
region_index = labels.index(region_name)

ts = region_timeseries(preds, region_of_vertex, region_index)

col_plot, col_brain = st.columns([3, 2])

with col_plot:
    st.subheader(f"{region_name}: activation over time")
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(ts)
    ax.set_xlabel("timestep")
    ax.set_ylabel("mean activation")
    ax.axhline(0, color="gray", linewidth=0.5)
    st.pyplot(fig)
    plt.close(fig)
    st.caption(
        f"mean predicted activation of {region_name} across '{title}'. "
        "above the gray line = more active than baseline"
    )

with col_brain:
    st.subheader("where it is")
    st.image(render_region_png(region_index),
             caption=f"{region_name} (lateral view; medial regions may be hidden)")