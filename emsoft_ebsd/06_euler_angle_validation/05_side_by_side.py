import kikuchipy as kp
import matplotlib.pyplot as plt
import numpy as np
from orix.vector import Vector3d
from orix.plot import IPFColorKeyTSL

real = kp.load("patterns.h5")
sim  = kp.load("../03_ebsd_pattern_simulation/Ni_EBSD_sim.h5")

ny, nx = real.xmap.shape

sym = real.xmap.phases[0].point_group
key = IPFColorKeyTSL(sym, direction=Vector3d.zvector())

rgb_real = key.orientation2color(real.xmap.rotations).reshape(ny, nx, 3)
rgb_sim  = key.orientation2color(sim.xmap.rotations).reshape(ny, nx, 3)

diff = np.abs(rgb_real - rgb_sim).max()
print(f"max colour difference: {diff:.6f}   (should be ~0)")

fig, ax = plt.subplots(1, 3, figsize=(16, 5))
ax[0].imshow(np.clip(rgb_real, 0, 1)); ax[0].set_title("real scan"); ax[0].axis("off")
ax[1].imshow(np.clip(rgb_sim, 0, 1));  ax[1].set_title("from simulated file"); ax[1].axis("off")
ax[2] = fig.add_subplot(133, projection="ipf", symmetry=sym)
ax[2].plot_ipf_color_key(); ax[2].set_title("IPF-Z key")
plt.tight_layout()
plt.savefig("outputs/ipf_real_vs_sim.png", dpi=180, bbox_inches="tight")