import h5py, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# f = h5py.File('Ni_EBSD_sim.h5', 'r')
f = h5py.File('Ni_EBSD_sim_NOISY.h5', 'r')
def find(name, obj):
    if isinstance(obj, h5py.Dataset) and obj.ndim >= 3:
        print(name, obj.shape, obj.dtype)
f.visititems(find)

d = f['EMData/EBSD/EBSDPatterns'][:]
print('patterns:', d.shape)

# angles.txt has a 2-line header ('eu', count) before the data - matches the
# angle file EMEBSD_ni.nml actually used to generate these patterns
angles = np.loadtxt('Nickel_100x100_angles.txt', skiprows=2)
GRID_NX = 100  # 100x100 Dream3D grid -> index = row*GRID_NX + col

# pick 3 patterns from 3 *different* grains (not just the first 3 indices,
# which are usually all the same grain - Dream3D repeats one orientation
# across every pixel inside a grain) by walking forward until the Euler
# angles actually change
chosen = [0]
for i in range(1, len(angles)):
    if not any(np.allclose(angles[i], angles[c], atol=1e-3) for c in chosen):
        chosen.append(i)
        if len(chosen) == 3:
            break

labels = []
for i in chosen:
    phi1, PHI, phi2 = angles[i]
    row, col = divmod(i, GRID_NX)
    labels.append(f'pixel ({row},{col})\nphi1={phi1:.1f} PHI={PHI:.1f} phi2={phi2:.1f}')

fig, ax = plt.subplots(1, 3, figsize=(15, 5))
for k, i in enumerate(chosen):
    ax[k].imshow(d[i], cmap='gray')
    ax[k].set_title(labels[k])
    ax[k].axis('off')
plt.tight_layout()
plt.savefig('Ni_patterns_NOISY.png', dpi=150, bbox_inches='tight')
print('wrote Ni_patterns_NOISY.png')
print('chosen pattern indices (3 different grains):', chosen)
