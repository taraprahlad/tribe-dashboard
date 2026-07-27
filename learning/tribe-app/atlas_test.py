import numpy as np
from nilearn import datasets

atlas = datasets.fetch_atlas_surf_destrieux()

# Save the vertex->region-index map for the left hemisphere
np.save("library/atlas_map_left.npy", np.array(atlas["map_left"]))

# Save the region names (decode bytes -> str), one per line
labels = [l.decode() if isinstance(l, bytes) else l for l in atlas["labels"]]
with open("library/atlas_labels.txt", "w") as f:
    f.write("\n".join(labels))

print("saved atlas to library/")