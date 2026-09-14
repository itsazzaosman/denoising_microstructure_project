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

---

## Each File Explained Simply

### **1. `environment.yml`** ← START HERE
**What it is:** A shopping list of packages your code needs.

**Example:**
```yaml
name: ebsd
dependencies:
  - python=3.11
  - pytorch::torch==2.12.1
  - pytorch::torchvision==0.27.1
  - numpy=2.2.6
  - scipy=1.15.3
  - pillow=12.3.0
  - matplotlib=3.10.9
```

**What it means:** "I need torch version 2.12.1, numpy 2.2.6, etc."

**When to edit:** Only if you need a NEW package (e.g., adding tensorboard)

---

### **2. `pyproject.toml`** ← Modern Python Standard
**What it is:** Python's standard way to describe a project.

**What it does:** Lists the same dependencies as `environment.yml` but in Python's standard format.

**Why both files?** 
- `environment.yml` = for conda (handles GPU/CUDA easily)
- `pyproject.toml` = for Python ecosystem compatibility

**You probably won't touch this.**

---

### **3. `conda-lock.yml`** ← The "Exact Blueprint" (Generate once)
**What it is:** A lock file that says "use EXACTLY this version, this build, this everything."

**Example:** Instead of "torch 2.12.1", it says "torch 2.12.1 build cu130_py311_5.8"

**Why?** Conda can resolve packages differently on different computers. This guarantees identical setup everywhere.

**How to generate (first time only):**
```bash
cd /project/community/aiosman/MAPS_DENOSING
bash generate-conda-lock.sh
```

**Status:** Created automatically by `generate-conda-lock.sh`

---

### **4. `SETUP.md`** ← Instructions for Others
**What it is:** A README that explains how to set up the environment.

**For others:** "Run `conda env create -f environment.yml` to get the exact setup"

**You don't need to read this unless you're stuck.**

---

### **5. `generate-conda-lock.sh`** ← One-Time Setup Script
**What it is:** A script that creates `conda-lock.yml`.

**When to run:** Once now, then only if you update `environment.yml`.

```bash
bash generate-conda-lock.sh
```

**What it does:**
1. Installs a tool called `conda-lock`
2. Creates `conda-lock.yml` (the exact blueprint)
3. Ready to commit to git

---

## Your Workflow (Day-to-Day)

### **Normal Training (Right Now)**
```bash
# Everything is already set up. Just run:
sbatch MAPS_DENOSING/src/train_model.sh
```

**That's it.** Nothing changes for you. The script now uses the `ebsd` environment instead of `diffusion`.

---

### **If You Need a New Package**

Example: You want to add `tensorboard` for visualization.

**Step 1:** Edit `MAPS_DENOSING/environment.yml`
```yaml
dependencies:
  - python=3.11
  - pytorch::torch==2.12.1
  - ... (other stuff)
  - tensorboard    # ← ADD THIS
```

**Step 2:** Regenerate the lock
```bash
bash MAPS_DENOSING/generate-conda-lock.sh
```

**Step 3:** Commit both files
```bash
git add MAPS_DENOSING/environment.yml MAPS_DENOSING/conda-lock.yml
git commit -m "Add tensorboard for visualization"
```

---

## For Someone Else to Run Your Code

They would do:
```bash
conda-lock install --name ebsd MAPS_DENOSING/conda-lock.yml
conda activate ebsd
cd MAPS_DENOSING/src
python train.py --epochs 50
```

**Result:** Identical environment to yours, guaranteed.

---

## Quick Reference: Do I Touch This File?

| File | Do I Edit? | Why? | When? |
|------|---|---|---|
| `environment.yml` | **YES** (sometimes) | Lists what to install | Only if adding packages |
| `pyproject.toml` | No | Python standard format | Never (auto-synced) |
| `conda-lock.yml` | No (auto-generated) | Exact blueprint | Generate once with script |
| `.gitignore` | No | Keeps git clean | Never |
| `SETUP.md` | No (reference only) | Instructions for others | Read if stuck |
| `generate-conda-lock.sh` | Run once | Creates lock file | Once now, then only if updating environment.yml |

---

## One-Time Setup: Do This Now

Generate the lock file (first time only):

```bash
cd /project/community/aiosman/MAPS_DENOSING
bash generate-conda-lock.sh
```

Then commit it:

```bash
git add conda-lock.yml
git commit -m "Lock exact conda versions for reproducibility"
```

**After that?** You're done. Keep using the cluster as normal. Everything else happens automatically.
