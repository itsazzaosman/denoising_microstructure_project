# Dependency Management

This project now uses **conda-lock + pyproject.toml** for robust, reproducible dependency management.

## What Changed

| File | Purpose | Status |
|------|---------|--------|
| `environment.yml` | Conda spec (human-readable) | ✓ Created |
| `pyproject.toml` | Python package metadata | ✓ Created |
| `conda-lock.yml` | Machine-generated lock file | Ready to generate |
| `generate-conda-lock.sh` | Script to generate lock file | ✓ Created |
| Conda environment | Renamed `diffusion` → `ebsd` | ✓ Done |
| `src/train_model.sh` | Updated to use `ebsd` env | ✓ Done |

## The Environment: `ebsd`

Your conda environment has been **cloned from `diffusion`** and renamed to **`ebsd`**.

**Verify it works:**
```bash
conda activate ebsd
python -c "import torch; print(f'torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"
```

**Location:** `~/miniconda3/envs/ebsd/`

## Next Step: Generate conda-lock.yml

When you're ready to lock the exact versions (so others can reproduce identically):

```bash
cd /project/community/aiosman/MAPS_DENOSING
bash generate-conda-lock.sh
```

This creates **`conda-lock.yml`** with every dependency pinned to exact version + build hash. 
Commit it to ensure reproducibility:
```bash
git add conda-lock.yml environment.yml pyproject.toml SETUP.md
git commit -m "Add reproducible dependency management with conda-lock"
```

## How to Use Each File

**For development (now):**
```bash
conda activate ebsd
cd MAPS_DENOSING/src
python train.py --epochs 10 --batch-size 32
```

**For others to reproduce (after conda-lock.yml is committed):**
```bash
conda-lock install --name ebsd conda-lock.yml
conda activate ebsd
# ... run scripts ...
```

**If you update dependencies:**
1. Edit `environment.yml` (e.g., add `- tensorboard`)
2. Regenerate lock: `bash generate-conda-lock.sh`
3. Commit both files

## Why This Setup?

| Aspect | Before | Now |
|--------|--------|-----|
| Reproducibility | Manual conda env creation | Locked via `conda-lock.yml` |
| Documentation | Implicit (env name only) | Explicit (`environment.yml`, `pyproject.toml`) |
| Dependency updates | Manual, error-prone | Tracked, versionable |
| Sharing code | Others had to guess packages | Exact reproduction guaranteed |
| CUDA handling | ✓ Works (conda native) | ✓ Still works |

## Key Packages

All versions match what's in your working `ebsd` environment:

```
torch==2.12.1+cu130           Deep learning
torchvision==0.27.1+cu130     Image utilities + pretrained models
numpy==2.2.6                  Numerical computing
scipy==1.15.3                 Scientific functions
pillow==12.3.0                Image I/O
matplotlib==3.10.9            Plotting
tqdm==4.68.3                  Progress bars
h5py                          HDF5 file support (auto-resolved)
```

Everything else (GPU libraries, transitive deps) is auto-resolved by conda.
