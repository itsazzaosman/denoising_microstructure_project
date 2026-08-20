import kikuchipy as kp

print("loading data...", flush=True)
s = kp.data.ni_gain(1, allow_download=True)   # (149, 200, 60, 60) = 29800 points, vs 4125 for nickel_ebsd_large
print("loaded", flush=True)

# The raw diffraction patterns (images) — analogous to real EBSPs
patterns = s.data          # shape: (149, 200, 60, 60)
print("patterns shape:", patterns.shape, flush=True)

# The already-indexed orientations for each point
xmap = s.xmap

# Convert those orientations to Euler angles (φ1, Φ, φ2), in radians by default
euler_angles = xmap.rotations.to_euler(degrees=True)

print(euler_angles.shape)   # (29800, 3)
print(euler_angles)

print("saving patterns.h5 ...", flush=True)
s.save("patterns.h5", overwrite=True)
print("wrote patterns.h5", flush=True)