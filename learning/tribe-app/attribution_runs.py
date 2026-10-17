"""attribution_runs.py: region-level attribution runs for the week 16 dashboard.

colab only, won't run locally: it needs the tribe v2 model loaded on a gpu
runtime. run it after: setup cell → restart → version check → model load.
it defines the week 15 helpers itself (copied from ablation_app.py):
make_region_forward, build_modality_mask, ablate_modalities, occlude_time,
and the region forward functions stg_forward, occ_forward and
control_forward.

regions (destrieux atlas, fsaverage5)
-------------------------------------
  stg        G_temp_sup-Lateral   speech anchor
  occipital  Pole_occipital       visual anchor
  control    S_postcentral        somatosensory, expected null; its values
                                  serve as the noise floor
each forward function pools the region's predicted response over vertices
and output time into one signed scalar per batch item.

runs
----
1. dorian gray, audio only, clip-mean baseline
   a. region × modality ablation (FeatureAblation; mask ids 0=text,
      1=audio, 2=video; slices 0:384 / 384:768 / 768:1152 of D=1152).
      also reproduces the week 15 stg row as an environment check.
   b. time-window occlusion (w=10, stride=5) for all three regions;
      200 input timesteps, ~0.3 s each.
2. tears of steel, real video, first 60 s
   a. the same ablation, clip-mean baseline. gate first: the video slice
      x_leaf[0, :, 768:1152] must be nonzero.
   b. the same ablation with a zero baseline. zero is how tribe encodes a
      missing modality, so this asks "what if this modality were absent?"

outputs (under LIB; copy to library/attribution/ locally)
---------------------------------------------------------
  dorian_modality_matrix.json        {region: {text, audio, video}}
  dorian_occlusion_{region}.npy      shape (200,)
  dorian_occlusion_params.json       {w, stride, n_timesteps}
  dorian_occlusion.png               the three curves, for the README / blog
  tears_modality_matrix.json         clip-mean baseline
  tears_modality_matrix_zero.json    zero baseline

reading the numbers
-------------------
- sign convention (captum): value = original − ablated/occluded. negative
  means removing that input RAISED the region's prediction.
- compare against the control region (the noise floor), not against zero.
- occlusion and ablation values are on different scales; don't plot them on
  a shared axis.
- the clip-mean baseline asks what the input's moment-to-moment variation
  contributes; the zero baseline asks what its presence contributes.
- the spike at the start of the occlusion curves appears in all three
  regions, so treat it as an edge effect.
- all results are single-clip, signed-pooled outputs of tribe's predicted
  response, not measurements of brain activity.

results and interpretation: see understanding-tribe.md, week 16 (F1–F4).
"""

# ======================================================================
# part i: setup
# ======================================================================

# imports
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from captum.attr import FeatureAblation, Occlusion
from nilearn import datasets

# drive mount + paths
from google.colab import drive
drive.mount('/content/drive')

LIB   = Path("/content/drive/MyDrive/tribe-app/library")  # drive backup of the library
CLIPS = LIB / "clips"          # where the tears of steel clip is saved
RAW   = Path("/content/raw")   # full downloads stay on local disk (disposable)
for d in (LIB, CLIPS, RAW):
    d.mkdir(parents=True, exist_ok=True)

# dorian audio download and clip: from 0 s, librivox intro INCLUDED.
# week 15's results and the occlusion timing depend on this exact clip.
!mkdir -p library
!wget -q --timeout=30 --tries=2 "https://archive.org/download/picture_doriangray_1012_librivox/pictureofdoriangray_13_wilde_64kb.mp3" -O library/dorian_audio_raw.mp3
!ffmpeg -y -i library/dorian_audio_raw.mp3 -ss 0 -t 60 -c copy library/dorian_audio_clip.mp3
!ls -la library/

# whisperx needs this for text alignment (also in the setup cell; harmless to repeat)
import nltk
nltk.download('punkt_tab')

# build the dorian batch and the attribution inputs
clip_path = "library/dorian_audio_clip.mp3"
events = model.get_events_dataframe(audio_path=clip_path)
loader = model.data.get_loaders(events=events, split_to_build="all")["all"]
batch = next(iter(loader)).to(model._model.device)

net = model._model
net.eval()
x_leaf = net.aggregate_features(batch).detach()    # (1, 200, 1152) = (B, T, D)
baseline = x_leaf.mean(dim=1, keepdim=True)        # clip-mean baseline, (1, 1, 1152)
D = x_leaf.shape[-1]
subject_id = batch.data.get("subject_id", None)    # read by the forward functions at call time
print(x_leaf.shape)

# destrieux atlas + regions (labels are plain str in this nilearn version, not bytes)
destrieux = datasets.fetch_atlas_surf_destrieux()
labels = destrieux["labels"]
regions = {
    "stg":            np.where(destrieux["map_left"] == labels.index("G_temp_sup-Lateral"))[0],
    "occipital pole": np.where(destrieux["map_left"] == labels.index("Pole_occipital"))[0],
    "control":        np.where(destrieux["map_left"] == labels.index("S_postcentral"))[0],
}
for name, idx in regions.items():
    print(name, len(idx))      # each should be a non-empty cluster of vertices

# ----------------------------------------------------------------------
# week 15 helpers
# ----------------------------------------------------------------------

# region-aware forward function: hand-runs the network tail and pools the
# region's prediction over vertices AND output time into one scalar per row
def make_region_forward(net, region_vertex_idx):
    def forward_func(x):
        h = x
        if hasattr(net, "temporal_smoothing"):
            h = net.temporal_smoothing(h.transpose(1, 2)).transpose(1, 2)
        if not net.config.linear_baseline:
            h = net.transformer_forward(h, subject_id)
        h = h.transpose(1, 2)
        if net.config.low_rank_head is not None:
            h = net.low_rank_head(h.transpose(1, 2)).transpose(1, 2)
        h = net.predictor(h, subject_id)
        preds_out = net.pooler(h)
        return preds_out[:, region_vertex_idx, :].mean(dim=(1, 2))
    return forward_func

stg_forward = make_region_forward(net, regions["stg"])
occ_forward = make_region_forward(net, regions["occipital pole"])
control_forward = make_region_forward(net, regions["control"])
print(stg_forward(x_leaf).shape)   # gate: want torch.Size([1])

# modality mask: id 0 = text (0:384), 1 = audio (384:768), 2 = video (768:1152)
def build_modality_mask(D):
    mask = torch.zeros(1, 1, D, dtype=torch.long)   # (1, 1, D), broadcasts over B and T
    mask[..., 384:768] = 1
    mask[..., 768:1152] = 2
    return mask

mask = build_modality_mask(D)
print(torch.unique(mask))                                         # want tensor([0, 1, 2])
print((mask == 0).sum(), (mask == 1).sum(), (mask == 2).sum())    # want 384 each

# time-window occlusion: one value per input timestep, shape (T,)
def occlude_time(forward_func, x, clip_mean, w, stride):
    occ = Occlusion(forward_func)
    attr = occ.attribute(
        x,
        sliding_window_shapes=(w, D),
        strides=(stride, D),
        baselines=clip_mean,
    )
    return attr.sum(dim=-1).squeeze(0)

# modality-channel ablation: every dim in a group shares one value, so read one per slice
def ablate_modalities(forward_func, x, clip_mean, modality_mask):
    fa = FeatureAblation(forward_func)
    attr = fa.attribute(
        x,
        feature_mask=modality_mask,
        baselines=clip_mean,
    )
    vec = attr[0, 0, :]
    return vec[0].item(), vec[384].item(), vec[768].item()   # text, audio, video

# everything part ii and iii need should exist now
needed = ["model", "LIB", "x_leaf", "baseline", "D", "subject_id", "stg_forward", "occ_forward",
          "control_forward", "build_modality_mask", "ablate_modalities", "occlude_time"]
print("missing:", [n for n in needed if n not in globals()] or "none")

# ======================================================================
# part ii: dorian results
# ======================================================================

FORWARDS = [("stg", stg_forward), ("occipital", occ_forward), ("control", control_forward)]

# dorian modality matrix
# reproduction check: stg should give text ≈ -0.0736, audio ≈ -0.1518, video 0
# (week 15 on cpu: -0.0738 / -0.1519 / 0)
matrix = {}
for name, ff in FORWARDS:
    t, a, v = ablate_modalities(ff, x_leaf, baseline, mask)
    matrix[name] = {"text": float(t), "audio": float(a), "video": float(v)}
    print(f"{name:10s} text {t:.4f}  audio {a:.4f}  video {v:.4f}")

with open(LIB / "dorian_modality_matrix.json", "w") as f:
    json.dump(matrix, f, indent=2)

# dorian occlusion curves + the params the app needs to put seconds on the x-axis
for name, ff in FORWARDS:
    curve = occlude_time(ff, x_leaf, baseline, w=10, stride=5)
    curve = curve.detach().cpu().numpy() if hasattr(curve, "detach") else np.asarray(curve)
    np.save(LIB / f"dorian_occlusion_{name}.npy", curve)
    print(name, curve.shape)    # want (200,)

with open(LIB / "dorian_occlusion_params.json", "w") as f:
    json.dump({"w": 10, "stride": 5, "n_timesteps": int(x_leaf.shape[1])}, f, indent=2)

# dorian occlusion plot (loads from drive, so it also confirms the saves)
fig, ax = plt.subplots(figsize=(10, 3))
for name in ["stg", "occipital", "control"]:
    ax.plot(np.load(LIB / f"dorian_occlusion_{name}.npy"), label=name)
ax.axhline(0, color="gray", lw=0.5)
ax.set_xlabel("input timestep (~0.3 s each)")
ax.legend()
plt.savefig(LIB / "dorian_occlusion.png", dpi=150, bbox_inches="tight")
plt.show()

# ======================================================================
# part iii: tears of steel results
# ======================================================================

# tears of steel clip: first 60 s, re-encoded the same way batch_expand.py's prepare() does
tos_raw = RAW / "tears_of_steel_raw.mp4"
tos_clip = CLIPS / "tears_of_steel_clip.mp4"
!wget -q "https://archive.org/download/tweakers006325/tweakers006325.mp4" -O {tos_raw}
!ffmpeg -y -ss 0 -i {tos_raw} -t 60 -c:v libx264 -preset veryfast -c:a aac {tos_clip}
print(tos_clip)

# rebuild the features from the video clip.
# NOTE: deliberately REUSES the dorian names (batch, x_leaf, baseline, D, subject_id).
# the forward functions read subject_id from the notebook when they run, so new
# names would leave them using dorian's values. don't "tidy" these into new names.
events = model.get_events_dataframe(video_path=str(tos_clip))
loader = model.data.get_loaders(events=events, split_to_build="all")["all"]
batch = next(iter(loader)).to(model._model.device)
net.eval()
x_leaf = net.aggregate_features(batch).detach()
baseline = x_leaf.mean(dim=1, keepdim=True)
D = x_leaf.shape[-1]
subject_id = batch.data.get("subject_id", None)

# gate: which modalities actually carry signal? video must be nonzero to continue
for name, sl in [("text", slice(0, 384)), ("audio", slice(384, 768)), ("video", slice(768, 1152))]:
    print(f"{name:6s} slice sum: {x_leaf[0, :, sl].abs().sum().item():.2f}")

# tears: clip-mean baseline matrix
mask = build_modality_mask(D)
matrix = {}
for name, ff in FORWARDS:
    t, a, v = ablate_modalities(ff, x_leaf, baseline, mask)
    matrix[name] = {"text": float(t), "audio": float(a), "video": float(v)}
    print(f"{name:10s} text {t:.4f}  audio {a:.4f}  video {v:.4f}")

with open(LIB / "tears_modality_matrix.json", "w") as f:
    json.dump(matrix, f, indent=2)

# tears: zero-baseline matrix. zero is tribe's own encoding for a missing modality,
# so this asks what each modality's PRESENCE contributes
zero = torch.zeros_like(baseline)
zmatrix = {}
for name, ff in FORWARDS:
    t, a, v = ablate_modalities(ff, x_leaf, zero, mask)
    zmatrix[name] = {"text": float(t), "audio": float(a), "video": float(v)}
    print(f"{name:10s} text {t:.4f}  audio {a:.4f}  video {v:.4f}")

with open(LIB / "tears_modality_matrix_zero.json", "w") as f:
    json.dump(zmatrix, f, indent=2)