**just some miscellaneous notes**

preds.shape --> (n_timesteps, n_vertices)
    - first number is time bc TRIBE predicts brain activity at multiple moments as the stimulus unfolds
    - second one is space -- there are ~70,000 vertices which each represent a specific location on the cortical surface, the number indicates how many spots on the brain are activated
this means that every row is the brain at one instant and each column is a single brain location across multiple instances

fMRI style predictions are typically normalized -- so roughly centered on zero, with positive meaning "activity above the baseline" and negative meaning "activity below the baseline." neural network outputs are generally 32-bit floating point numbers where in this case, each float32 value is the predicted activation at that point.

fsaverage5 is the standard cortical surface template -- since every brain is different, a brain has to be mathematically transformed to fit the template and this is what allows the vertices to map onto parts of the brain correctly. additionally, it's worthing that fsaverage has a specific resolution and that is 20484 vertices, it's lighter to compute and visualize.

tribe automatically converts from tensors to arrays so it the outputs can clearly be given to libraries like nilearn that use array-like data.

**videos:**

tribe doesn't rawdog vids, it breaks the stimulus down into time-aligned events table across the three modalities. since we have a "movie" and "predicted brain activity over time", the preds output has a time dimension.

**the product of more messing around**
you can change the color map from sequential to diverging -- diverging is better for me because it shows the positive and negative activations as opposed to a spectrum if that makes any sense? you can change the centering as well so that the diverging colormap is meaningful, the default render is a gray brain but if you change normpercentile to None it should become white (this is because plotbrain applies normalization before vmin/vmax).

for getting both hemispheres to render at once, you need to play with the plot_surf method instead of plot_timesteps -- plot_surf renders a single timestep while plot_timesteps renders a sequence. with plot surf you can change the axes and the views to render both hemispheres. the code below is what I used!

```python
import matplotlib.pyplot as plt

fig, axd = plt.subplot_mosaic(
    [["left", "right"] ["medial_left", "medial_right"]],
    figsize(12, 10)
)

plotter.plot_surf(
    preds=[0],
    axes=axd,
    cmap="coolwarm",
    symmetric_car="True",
    norm_percentile=None,
)
```