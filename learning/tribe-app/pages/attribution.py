"""tribe attribution (page of app.py)

what drives the model's predictions? two methods, both run offline in colab
(colab/attribution_runs.py) and loaded here as precomputed results:
  1. time-window occlusion on dorian gray: which moments of the input matter
  2. region × modality ablation: which input modality each region leans on"""

import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from tribe_shared import load_attribution_json, load_attribution_npy

REGIONS = ["stg", "occipital", "control"]
MODALITIES = ["text", "audio", "video"]

st.title("tribe attribution")
st.write(
    "these results are TRIBE's *predicted* response for single clips, not brain "
    "measurements. compare every number against the control region (somatosensory "
    "cortex, expected to be uninvolved): anything about its size is noise."
)

# ----------------------------------------------------------------------
# 1. occlusion over time (dorian gray)
# ----------------------------------------------------------------------
st.header("1. which moments matter? (occlusion, dorian gray)")

params = load_attribution_json("dorian_occlusion_params.json")   # {w, stride, n_timesteps}
curves = {r: load_attribution_npy(f"dorian_occlusion_{r}.npy") for r in REGIONS}

seconds = np.arange(params["n_timesteps"]) * (60 / params["n_timesteps"])

fig, ax = plt.subplots(figsize=(10, 3.2))
for r in REGIONS:
    ax.plot(seconds, curves[r], label=r)
ax.axhline(0, color="gray", lw=0.5)

ax.axvspan(
    0,
    params["w"] * (60 / params["n_timesteps"]),
    color="gray",
    alpha=0.15,
)

ax.axvline(33.3, color="black", lw=0.8, ls="--")
ax.text(33.8, ax.get_ylim()[1] * 0.85, "real narration begins (~33 s)", fontsize=8)
ax.set_xlabel("time in clip (s)")
ax.set_ylabel("original − occluded")
ax.legend()
st.pyplot(fig)
plt.close(fig)

st.caption(
    "each point: how much the region's prediction changes when that moment of input is "
    "replaced with the clip average. negative = removing it RAISED the prediction. "
    "after the librivox intro, stg and occipital drop while control stays flat: the "
    "effect is region-selective. the shaded spike at the start appears in all three "
    "regions, so it's an edge effect, not a finding."
)

# ----------------------------------------------------------------------
# 2. region × modality ablation
# ----------------------------------------------------------------------
st.header("2. which input does each region lean on? (modality ablation)")

RUNS = {
    "dorian gray: audio only, clip-mean baseline": "dorian_modality_matrix.json",
    "tears of steel: real video, clip-mean baseline": "tears_modality_matrix.json",
    "tears of steel: real video, zero baseline (modality removed)": "tears_modality_matrix_zero.json",
}
NOTES = {
    "dorian_modality_matrix.json":
        "audio clearly matters for both stg and occipital, with the same sign: consistent with a "
        "shared component rather than one-region-one-modality. video is 0 because the clip has none.",
    "tears_modality_matrix.json":
        "no clean dissociation: occipital is most sensitive to audio, stg to video.",
    "tears_modality_matrix_zero.json":
        "zero is how tribe encodes a missing modality. stg's video effect survives both baselines "
        "(the most robust result here); occipital barely moves when any single modality is removed, "
        "which points to redundancy across modalities.",
}

run_label = st.selectbox("run", list(RUNS))
matrix = load_attribution_json(RUNS[run_label])          # {region: {text, audio, video}}
df = pd.DataFrame(matrix).T[MODALITIES]                  # rows = regions, cols = modalities

noise_floor = df.loc["control"].abs().max()

# grouped bars: one group per region, one bar per modality
fig, ax = plt.subplots(figsize=(8, 3.5))
x = np.arange(len(REGIONS))
width = 0.25
for i, m in enumerate(MODALITIES):
    ax.bar(x + (i - 1) * width, df.loc[REGIONS, m], width, label=m)
ax.axhspan(-noise_floor, noise_floor, color="gray", alpha=0.15, label="noise floor (control)")
ax.axhline(0, color="gray", lw=0.5)
ax.set_xticks(x, REGIONS)
ax.set_ylabel("original − ablated")
ax.legend(fontsize=8)
st.pyplot(fig)
plt.close(fig)

st.caption(NOTES[RUNS[run_label]])
md = "| region | text | audio | video |\n|---|---|---|---|\n"
for region, r in df.iterrows():
    md += f"| {region} | {r['text']:.4f} | {r['audio']:.4f} | {r['video']:.4f} |\n"
st.markdown(md)

with st.expander("how to read this / caveats"):
    st.markdown(
        "- **sign:** value = original − ablated. negative means removing that modality *raised* the prediction.\n"
        "- **baselines:** clip-mean asks what a modality's moment-to-moment variation contributes; "
        "zero asks what its presence contributes.\n"
        "- **pooling:** each region is one signed average over all its vertices and timesteps, so "
        "opposite effects can cancel.\n"
        "- **scope:** single clips, one model, the average subject. treat these as observations about "
        "TRIBE, not claims about brains.\n"
        "- **scale:** occlusion and ablation values aren't on the same scale, so don't compare across sections."
    )