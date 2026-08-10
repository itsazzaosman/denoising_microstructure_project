import h5py, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

f = h5py.File('Ni_EBSD.h5', 'r')
def find(name, obj):
    if isinstance(obj, h5py.Dataset) and obj.ndim >= 3:
        print(name, obj.shape, obj.dtype)
f.visititems(find)

d = f['EMData/EBSD/EBSDPatterns'][:]
print('patterns:', d.shape)
labels = ['(0, 0, 0) cube', '(30, 45, 10)', '(0, 54.7, 45) <111>']
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for i in range(3):
    ax[i].imshow(d[i], cmap='gray')
    ax[i].set_title(labels[i])
    ax[i].axis('off')
plt.tight_layout()
plt.savefig('Ni_patterns.png', dpi=150, bbox_inches='tight')
print('wrote Ni_patterns.png')
