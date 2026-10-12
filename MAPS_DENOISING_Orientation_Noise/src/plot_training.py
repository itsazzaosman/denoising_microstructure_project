#!/usr/bin/env python3
"""
Training curves: training vs validation loss, and the validation errors.

train.py calls draw_curves() after every epoch to write
<run folder>/training_curves.png. Run this file directly to redraw a run from
its history.json, or to put several runs (e.g. the three seeds) on one figure.

Panels
  loss              training and validation loss, same formula (denoise_loss)
  loss components   the validation loss split into its four terms
                    (one run), or the model-selection score (several runs)
  ALL               validation median error over all pixels, vs the noisy input
  BOUNDARY          validation median error on grain-boundary pixels
  CORRUPTED         validation median error on the misindexed pixels
  gate              precision and recall of the misindexing detector

The dotted vertical line is the epoch with the best validation score, i.e. the
epoch saved as best.pth.

The training loss is averaged over the epoch while the weights are still
changing, and on fresh random noise; the validation loss is computed once at
the end of the epoch on the fixed validation set. A training curve sitting a
little above the validation curve early on is therefore expected, not a sign
of a bug. The warning sign is the opposite: validation loss rising while the
training loss keeps falling.

Usage
-----
    python plot_training.py ../checkpoints/runs/blind_learned_seed1
    python plot_training.py ../checkpoints/runs/blind_learned_seed{1,2,3} \\
        --out ../figures/training_seeds.png
"""

import argparse
import json
from pathlib import Path

import numpy as np

INK, INK_2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
RUN_STYLES = ["-", "--", ":", "-."]


def _val(h, key):
    return h["val"].get(key, float("nan"))


def _best_epoch(history):
    scored = [h for h in history if "score" in h["val"]]
    return min(scored, key=lambda h: h["val"]["score"])["epoch"] if scored else None


def draw_curves(runs, path):
    """runs: {name: history list}. Writes one figure to `path`."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = {k: v for k, v in runs.items() if v}
    if not runs:
        return
    single = len(runs) == 1
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "text.color": INK,
                         "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
                         "axes.edgecolor": MUTED})
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.2))
    ax = axes.ravel()

    for r, (name, hist) in enumerate(runs.items()):
        ep = [h["epoch"] for h in hist]
        colour = SERIES[r % len(SERIES)]
        tag = "" if single else f"{name} "

        # loss: train vs val
        if single:
            ax[0].plot(ep, [h["train_loss"] for h in hist], color=SERIES[0], lw=2, label="training")
            if "loss" in hist[0]["val"]:
                ax[0].plot(ep, [_val(h, "loss") for h in hist], color=SERIES[1], lw=2, label="validation")
        else:
            ax[0].plot(ep, [h["train_loss"] for h in hist], color=colour, lw=2, ls="--",
                       label=f"{tag}training")
            if "loss" in hist[0]["val"]:
                ax[0].plot(ep, [_val(h, "loss") for h in hist], color=colour, lw=2,
                           label=f"{tag}validation")

        # components (one run) or the selection score (several)
        if single and "loss_parts" in hist[0]["val"]:
            for k, part in enumerate(["main", "gate", "refine", "replace"]):
                ax[1].plot(ep, [h["val"]["loss_parts"].get(part, float("nan")) for h in hist],
                           color=SERIES[k], lw=2, label=part)
        elif not single:
            ax[1].plot(ep, [_val(h, "score") for h in hist], color=colour, lw=2, label=name)

        for i, key in ((2, "all"), (3, "bnd"), (4, "bad")):
            ax[i].plot(ep, [_val(h, key) for h in hist], color=colour, lw=2,
                       label="model" if single else name)

        ax[5].plot(ep, [_val(h, "gate_precision") for h in hist], color=SERIES[0] if single else colour,
                   lw=2, ls="-", label=f"{tag}precision")
        ax[5].plot(ep, [_val(h, "gate_recall") for h in hist], color=SERIES[1] if single else colour,
                   lw=2, ls="-" if single else "--", label=f"{tag}recall")

        best = _best_epoch(hist)
        if best is not None:
            for a in ax:
                a.axvline(best, color=colour if not single else MUTED, ls=":", lw=1.2)

    # Noisy-input reference: one grey line when every run shares a validation
    # set (the seeds), one line per run otherwise.
    inputs = {name: h[0]["val"]["in_all"] for name, h in runs.items() if "in_all" in h[0]["val"]}
    if inputs and len({round(v, 9) for v in inputs.values()}) == 1:
        ax[2].axhline(next(iter(inputs.values())), color=MUTED, ls="--", lw=1.2, label="noisy input")
    else:
        for r, (name, v) in enumerate(inputs.items()):
            ax[2].axhline(v, color=SERIES[list(runs).index(name) % len(SERIES)], ls="--", lw=1,
                          label=f"{name} input")
    first = next(iter(runs.values()))
    last_epoch = max(h[-1]["epoch"] for h in runs.values())

    titles = ["loss (training vs validation)",
              "validation loss components" if single else "validation score (lower = better)",
              "validation ALL, median error (deg)", "validation BOUNDARY, median error (deg)",
              "validation CORRUPTED, median error (deg)", "misindexing gate"]
    for a, t in zip(ax, titles):
        a.set_xlim(0.5, last_epoch + 0.5)       # same epoch range on every panel
        a.set_title(t, color=INK, loc="left")
        a.set_xlabel("epoch")
        a.grid(color=GRID, lw=0.8)
        a.set_axisbelow(True)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
        if a.get_legend_handles_labels()[0]:
            a.legend(frameon=False, fontsize=8)
    for i in (0, 1, 2, 3, 4):
        # Log scale only where there is something to show: old runs have no
        # validation loss, and scatter-only runs have no misindexed pixels.
        ys = [y for line in ax[i].get_lines() for y in line.get_ydata()
              if not line.get_linestyle() == ":" and np.isfinite(y) and y > 0]
        if ys:
            ax[i].set_yscale("log")
        else:
            ax[i].text(0.5, 0.5, "not recorded for this run", transform=ax[i].transAxes,
                       ha="center", va="center", color=MUTED)
    ax[5].set_ylim(0, 1.02)
    # Plain numbers (0.42, not 4.2x10^-1) and whole epochs on every axis.
    from matplotlib.ticker import FuncFormatter, MaxNLocator, NullFormatter
    plain = FuncFormatter(lambda v, _: f"{v:g}")
    for a in ax:
        a.xaxis.set_major_locator(MaxNLocator(integer=True))
        if a.get_yscale() == "log":
            a.yaxis.set_major_formatter(plain)
            lo, hi = a.get_ylim()
            # Label minor ticks only on a narrow range, where they are the only
            # labels; across several decades they would pile up.
            a.yaxis.set_minor_formatter(plain if hi / lo < 10 else NullFormatter())

    if single:
        best = _best_epoch(first)
        name = next(iter(runs))
        fig.suptitle(f"{name}" + (f"   (best epoch {best}, dotted line)" if best else ""),
                     x=0.01, ha="left", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+", help="run folders (holding history.json) or history.json files")
    p.add_argument("--out", default=None,
                   help="output image; default <run>/training_curves.png for one run, "
                        "./training_comparison.png for several")
    args = p.parse_args()

    runs, folders = {}, []
    for r in args.runs:
        path = Path(r)
        path = path / "history.json" if path.is_dir() else path
        runs[path.parent.name] = json.loads(path.read_text())
        folders.append(path.parent)
    out = args.out or (folders[0] / "training_curves.png" if len(runs) == 1
                       else Path("training_comparison.png"))
    draw_curves(runs, out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
