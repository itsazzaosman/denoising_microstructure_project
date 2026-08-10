import kikuchipy as kp
import matplotlib.pyplot as plt
import numpy as np
from orix.vector import Vector3d
from orix.plot import IPFColorKeyTSL

print("loading data...", flush=True)
s = kp.load("patterns.h5")
# s = kp.load("../03_ebsd_pattern_simulation/Ni_EBSD_sim.h5")
xmap = s.xmap
print("the data is loaded")
# Nickel is cubic (Fm-3m / m-3m point group) — get the symmetry
phase = xmap.phases[0]
symmetry = phase.point_group

# Choose the sample direction to project onto (Z is standard for IPF-Z maps)
ipf_key = IPFColorKeyTSL(symmetry, direction=Vector3d.zvector())

# Get RGB colors for every orientation in the map
rgb = ipf_key.orientation2color(xmap.rotations)

# Reshape colors back into the (55, 75) spatial grid to plot as an image
rgb_map = rgb.reshape(xmap.shape + (3,))   # xmap.shape should be (55, 75)
# rgb_map = rgb.reshape(55, 75, 3)

fig = plt.figure(figsize=(10, 6))

ax1 = fig.add_subplot(121)
ax1.imshow(rgb_map)
ax1.set_title("IPF-Z map")
ax1.axis("off")

ax2 = fig.add_subplot(122, projection="ipf", symmetry=symmetry)
ax2.plot_ipf_color_key()
ax2.set_title("IPF-Z color key")

plt.tight_layout()
# plt.savefig("outputs/ipf_z_map_with_key.png", dpi=200)
plt.savefig("outputs/ipf_z_map_with_key_sim.png", dpi=200)

plt.show()