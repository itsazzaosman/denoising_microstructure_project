#!/usr/bin/env python3
"""
STEP 6 -- diagnose why simulated and real patterns disagree.

Phase 1 (free): tests whether a simple flip/rotation of the simulated patterns
brings them into agreement. EMsoft and kikuchipy do not always agree on which
way is up, and if that is the only problem the geometry is already correct.

Scores every candidate transform by normalised cross-correlation (NCC) averaged
over many scan points. NCC of 1.0 is identical, 0.0 is unrelated.

Run from ~/EMsoftData_work/06_euler_angle_validation:
    python3 06_diagnose_match.py
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import h5py
import kikuchipy as kp

p = argparse.ArgumentParser()
p.add_argument("--real", default="patterns.h5")
p.add_argument("--sim", default="../03_ebsd_pattern_simulation/Ni_EBSD_sim.h5")
p.add_argument("--n", type=int, default=200, help="how many scan points to test")
p.add_argument("--out", default="outputs/diagnose_match.png")
args = p.parse_args()


def ncc(a, b):
    """Normalised cross-correlation between two images."""
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    a -= a.mean()
    b -= b.mean()
    da, db = a.std(), b.std()
    if da < 1e-12 or db < 1e-12:
        return 0.0
    return float(np.mean(a * b) / (da * db))


# ----------------------------------------------------------------------
print(f"Loading real scan: {args.real}")
s = kp.load(args.real)
ny, nx = s.axes_manager.navigation_shape[::-1]
s.remove_static_background()
s.remove_dynamic_background()
real = np.asarray(s.data, dtype=np.float32)

print(f"Loading simulated: {args.sim}")
with h5py.File(args.sim, "r") as f:
    sim = f["EMData/EBSD/EBSDPatterns"][:]
sim = np.squeeze(sim).astype(np.float32)
print(f"  real {real.shape}, simulated {sim.shape}")

# sample points spread across the map
rng = np.random.default_rng(0)
idx = np.sort(rng.choice(ny * nx, min(args.n, ny * nx), replace=False))

# ----------------------------------------------------------------------
# Candidate transforms of the simulated pattern
# ----------------------------------------------------------------------
transforms = {
    "identity":            lambda p: p,
    "flip up-down":        lambda p: p[::-1, :],
    "flip left-right":     lambda p: p[:, ::-1],
    "rotate 180":          lambda p: p[::-1, ::-1],
    "transpose":           lambda p: p.T,
    "rotate 90 cw":        lambda p: np.rot90(p, -1),
    "rotate 90 ccw":       lambda p: np.rot90(p, 1),
    "transpose + flip ud": lambda p: p.T[::-1, :],
}

print(f"\nScoring {len(transforms)} transforms over {len(idx)} scan points\n")
results = {}
for name, fn in transforms.items():
    scores = []
    for i in idx:
        r, c = divmod(i, nx)
        scores.append(ncc(real[r, c], fn(sim[i])))
    scores = np.array(scores)
    results[name] = (scores.mean(), scores.std(), np.median(scores))

print(f"{'transform':<22} {'mean NCC':>10} {'median':>10} {'std':>8}")
print("-" * 54)
for name, (m, sd, med) in sorted(results.items(), key=lambda kv: -kv[1][0]):
    print(f"{name:<22} {m:>10.4f} {med:>10.4f} {sd:>8.4f}")

best_name, (best_mean, _, _) = max(results.items(), key=lambda kv: kv[1][0])

print(f"\nBest: '{best_name}' with mean NCC {best_mean:.4f}")
if best_mean > 0.35:
    print("  -> Strong match. The geometry is right; this is just an axis convention.")
elif best_mean > 0.15:
    print("  -> Weak match. Partly right, but the pattern centre is probably off too.")
else:
    print("  -> No match. The problem is not a flip: suspect the Euler convention,")
    print("     the pattern centre, or the orientation ordering.")

# ----------------------------------------------------------------------
# Figure: real vs simulated under the best transform
# ----------------------------------------------------------------------
picks = [(9, 12), (18, 37), (27, 25), (36, 50)]
fn = transforms[best_name]
fig, ax = plt.subplots(3, len(picks), figsize=(3.1 * len(picks), 9.4))
for j, (r, c) in enumerate(picks):
    i = r * nx + c
    ax[0, j].imshow(real[r, c], cmap="gray")
    ax[0, j].set_title(f"real ({r},{c})", fontsize=10)
    ax[1, j].imshow(sim[i], cmap="gray")
    ax[1, j].set_title("simulated, as stored", fontsize=10)
    ax[2, j].imshow(fn(sim[i]), cmap="gray")
    ax[2, j].set_title(f"simulated, {best_name}\nNCC={ncc(real[r,c], fn(sim[i])):.3f}",
                       fontsize=9)
    for k in range(3):
        ax[k, j].axis("off")
plt.tight_layout()
plt.savefig(args.out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nwrote {args.out}")