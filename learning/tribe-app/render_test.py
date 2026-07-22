"""stage 1: proving we can render a tribe prediction locally with nilearn.

google colab was being a bitch so we're doing this without a gpu. also without streamlit but that's temporary. 

run it by going inside the tribe-app folder and: python render_test.py"""

import numpy as np
from nilearn import datasets, plotting

#loading the prediction saved from colab
preds = np.load("preds.npy")
print("loaded preds with shape:", preds.shape)

#picking a timestep to look at
#preds[t] is the whole brain at moment t (flat array of 20484 values)
timestep = 0
one_timestep = preds[timestep]

#split into left and right hemispheres
#fsaverage5 has 10424 ertices per hemisphere so the first half of the array is the left hemisphere
#and the second half is the right
n_per_hemi = 10242
left_data = one_timestep[:n_per_hemi]
right_data = one_timestep[n_per_hemi:]
print("left hemi:", left_data.shape, "right-hemi:", right_data.shape)

#loading fsaverage5 surface meshes
fsaverage = datasets.fetch_surf_fsaverage(mesh="fsaverage5")

#render one hemisphere as a quick proof it works
"""plot_surf_stat_map draws data on a surface mesh. we can use the left hemisphere's
inflated mesh (easier to see than the folded surface) and the left data.
`view_surf` gives an interactive 3D view (like Week 4's view_img)."""
view = plotting.view_surf(
    surf_mesh = fsaverage["infl_left"],
    surf_map = left_data,
    cmap = "coolwarm",
    symmetric_cmap = True,
    title = f"left hemisphere, timestep {timestep}",
)

view.save_as_html("brain_render.html")
print("successfully saved render to brain_render.html -- try opening it in your browser!")