"""messing around with integrated gradients! this makes a comparison between the peak activations of the 
stg at timestep 30 between the vanilla and ig saliency maps. i ran this is google colab, you
can't run it locally."""

import matplotlib.pyplot as plt

# integrated gradients curve -- collapse (1, 200, 1152)
ig_curve = attributions.abs().sum(dim = 2).squeeze(0).detach().cpu().numpy()

# vanilla saliency
xv = x_leaf.clone().requires_grad_(True)
s = forward_func(xv)
s.backward()
van_curve = xv.grad.abs().sum(dim = 2).squeeze(0).detach().cpu().numpy()

# normalize each to its own peak
ig_n = ig_curve / ig_curve.max()
van_n = van_curve / van_curve.max()

# make the plot
fig, ax = plt.subplots(figsize = (10, 4))
ax.plot(van_n, label = "vanilla saliency", alpha = 0.7)
ax.plot(ig_n, label = "integrated gradients", alpha = 0.7)
ax.set_xlabel("input timestep (0-199)")
ax.set_ylabel("peak normalized importance")
ax.set_title("STG @ t = 30 -- vanilla v.s. IG")
ax.legend()
plt.show()
plt.close(fig)