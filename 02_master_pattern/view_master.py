import h5py, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

f = h5py.File('Ni_master_hires.h5', 'r')
d = f['EMData/EBSDmaster/mLPNH'][:]
print('dataset shape:', d.shape)
img = np.squeeze(d)
while img.ndim > 2:
    img = img.sum(axis=0)
plt.figure(figsize=(7, 7))
plt.imshow(img, cmap='gray')
plt.axis('off')
plt.tight_layout()
plt.savefig('Ni_master.png', dpi=150, bbox_inches='tight')
print('wrote Ni_master.png')
