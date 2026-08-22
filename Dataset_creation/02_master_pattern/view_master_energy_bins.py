import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'DejaVu Sans'

f = h5py.File('Ni_master_hires.h5', 'r')
d = f['EMData/EBSDmaster/mLPNH'][:]   # (numset, numEbins, npx, npx)
EkeVs = f['EMData/EBSDmaster/EkeVs'][:]

d = d[0]  # single atom site (numset == 1)
numEbins = d.shape[0]
print('energy bins:', numEbins, 'keV values:', EkeVs)

# shared intensity scale across all bins, so brightness is comparable
# panel-to-panel instead of each bin auto-stretching its own contrast
vmin, vmax = np.percentile(d, [1, 99])

ncols = 4
nrows = int(np.ceil(numEbins / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
axes = np.atleast_1d(axes).ravel()

for i in range(numEbins):
    ax = axes[i]
    ax.imshow(d[i], cmap='gray', vmin=vmin, vmax=vmax)
    ax.set_title(f'{EkeVs[i]:.0f} keV')
    ax.axis('off')

for ax in axes[numEbins:]:
    ax.axis('off')

fig.suptitle('Ni master pattern (mLPNH) per energy bin')
plt.tight_layout()
plt.savefig('Ni_master_energybins.png', dpi=150, bbox_inches='tight')
print('wrote Ni_master_energybins.png')
