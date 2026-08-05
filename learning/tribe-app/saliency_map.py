"""this is a saliency map for the stg region of the brain that processes language (because
we saw a clear difference in its predicted activation across stimuli)! For a single speech-rich
clip, you should be able to get a saliency curve for the stimulus over time -- you can figure out
which input moments drove stg activation at one timestep?

note: this runs in colab against the loaded model"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from nilearn import datasets

# picking the clip and target timestep!
clip_path = "library/dorian_audio_clip.mp3"
target_t = 35

# get stg's vertex indices
destrieux = datasets.fetch_atlas_surf_destrieux()
labels = destrieux["labels"]
stg_label = "Pole_occipital"
stg_code = labels.index(stg_label)
stg_vertices = np.where(destrieux["map_left"] == stg_code)[0]

# first batch
events = model.get_events_dataframe(audio_path = clip_path)
loader = model.data.get_loaders(events = events, split_to_build = "all")["all"]
batch = next(iter(loader)).to(model._model.device)

net = model._model
net.eval()
subject_id = batch.data.get("subject_id", None)

x_leaf = net.aggregate_features(batch)
x_leaf = x_leaf.detach().requires_grad_(True)
print("B x.is_leaf:", x.is_leaf)

h = x_leaf
if hasattr(net, "temporal_smoothing"):
  h = net.temporal_smoothing(h.transpose(1, 2)).transpose(1,2)
if not net.config.linear_baseline:
  h = net.transformer_forward(h, subject_id)
h = h.transpose(1, 2)
if net.config.low_rank_head is not None:
  h = net.low_rank_head(h.transpose(1, 2)).transpose(1, 2)
h = net.predictor(h, subject_id)
preds_out = net.pooler(h)

# stg scalar
stg_scalar = preds_out[0, stg_vertices, target_t].mean()

# backward pass
print("C stg_scalar.requires_grad:", stg_scalar.requires_grad)
print("D stg_scalar.grad_fn:", stg_scalar.grad_fn)
stg_scalar.backward()
grad = x_leaf.grad
assert grad is not None, "no gradient reached x leaf -- graph got cut somewhere"

#reduce to curve over input time
saliency_curve = grad[0].norm(dim = 1).detach().cpu().numpy()

# make the graph!
seconds = np.linspace(0, 60, saliency_curve.shape[0])
fig, ax = plt.subplots()
ax.plot(seconds, saliency_curve)
ax.set_xlabel("time in clip (s)")
ax.set_ylabel("occ saliency (grad magnitude)")
ax.set_title(f"occ saliency at t={target_t}")
plt.show()
plt.close(fig)

