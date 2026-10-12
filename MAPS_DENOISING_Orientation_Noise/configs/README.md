# Experiment configs

One YAML file per experiment. The file name is the experiment's name.

| folder | one file per | read by | output |
|---|---|---|---|
| `noise/` | test set | `add_ebsd_noise.py` (`noise:` section) | `datasets/noise_<name>/` |
| `train/` | training run | `train.py` (`train:` section) | `checkpoints/runs/<name>/` |

Inside a section, each key is the script's option name with `-` written as
`_` (`--scatter-deg` → `scatter_deg`). A key the script does not know is an
error, so a typo cannot silently fall back to a default. A flag typed on the
command line overrides the file.

## Running

From `/project/community/aiosman`:

```bash
make noise NOISE=scatter5                  # build datasets/noise_scatter5 from noise/scatter5.yaml
CONFIG=scatter7 sbatch MAPS_DENOISING_Orientation_Noise/src/train_model.sh --resume auto
make infer RUN=scatter7 NOISE=scatter7     # predictions from checkpoints/runs/scatter7
make score-all NOISE=scatter7
```

`CONFIG` also sets the checkpoint folder (`checkpoints/runs/<name>`) and,
through the file's `test_set:`, which maps are scored when training ends.
Resubmitting the same `CONFIG` with `--resume auto` continues that run after a
preemption.

## Rules

- **New experiment, new file.** Never edit a config once its run has started.
  Copy it under a new name instead (`blind_seed1.yaml`, `blind_seed2.yaml`, …).
  Otherwise the file no longer describes the results next to it.
- **`seed` does not fix everything.** It fixes the train/validation split, the
  validation noise and the initial weights. The training noise is drawn fresh
  every epoch without a seed, so two runs of one file differ slightly. That is
  what the 3-seed repeats in the plan measure.
- Hardware choices (`num_workers`, `no_amp`) are left to the job scripts.

## Where the code version is recorded

Every result records the git commit that produced it, with `-dirty` added
when `src/` or `configs/` had uncommitted changes. Commit before a run that
will go in the paper, so the hash alone identifies the code.

| result | where the commit is |
|---|---|
| `best.pth`, `last.pth` | `ckpt["provenance"]` |
| `history.json` | `commit` in every epoch entry |
| run folder | `run_info.json`: one entry per start or resume, with the config and all settings |
| test set folder | `provenance.json`: one entry per run of `add_ebsd_noise.py` |
| model outputs in `mtex_out_<set>/` | `<method>.provenance.json`, including the checkpoint's own commit |
| `score_dataset.py --csv` | a `commit` column, plus `<csv>.provenance.json` with the full command (including `--metric`) |

The `scatter5/7/9` runs and test sets were made before this existed, so their
code version is unknown. Their configs were rebuilt from the settings saved in
the checkpoints and checked against the files on disk.
