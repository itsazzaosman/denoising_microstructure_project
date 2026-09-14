# Environment Setup

This project uses conda for dependency management with `conda-lock` for reproducibility.

## Quick Start

### 1. Clone or recreate the environment

**Option A: From the project (fastest)**
```bash
conda create --name ebsd --clone diffusion  # if diffusion still exists
# OR
conda env create -f environment.yml --name ebsd
```

**Option B: From conda-lock (most reproducible)**
```bash
# First install conda-lock:
conda install -c conda-forge conda-lock

# Then create the exact locked environment:
conda-lock install --name ebsd conda-lock.yml
```

### 2. Activate the environment
```bash
conda activate ebsd
```

### 3. Verify the installation
```bash
python -c "import torch; print(torch.__version__)"
python -c "import torchvision; print(torchvision.__version__)"
```

## What's What

- **`environment.yml`** — Human-readable conda specification. Pinned to specific versions but allows conda to resolve transitive dependencies. Use this for development.

- **`pyproject.toml`** — Python package metadata (project name, version, core dependencies). Use this if you want to install the project via `pip install -e .` (currently not set up as an installable package).

- **`conda-lock.yml`** — Machine-generated lock file. Every dependency (including transitive ones) is pinned to an exact version and build hash. Guarantees bit-for-bit reproducibility. Regenerate with:
  ```bash
  conda-lock lock --file environment.yml
  ```

## Updating Dependencies

To add a new package (e.g., `tensorboard`):

1. Edit `environment.yml` — add `- tensorboard` to the dependencies list
2. Update the lock file:
   ```bash
   conda-lock lock --file environment.yml
   ```
3. Commit both files

## Key Packages

| Package | Version | Purpose |
|---------|---------|---------|
| torch | 2.12.1 | PyTorch deep learning |
| torchvision | 0.27.1 | Image utilities |
| numpy | 2.2.6 | Numerical computing |
| scipy | 1.15.3 | Scientific computing |
| pillow | 12.3.0 | Image I/O |
| matplotlib | 3.10.9 | Plotting |
| h5py | — | HDF5 file support (auto-resolved) |

## GPU Support

The environment includes CUDA 13.1 toolkit via the `pytorch::pytorch-cuda=13.1` dependency.
To verify GPU support:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```
