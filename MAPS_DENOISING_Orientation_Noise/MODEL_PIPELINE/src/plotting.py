#!/usr/bin/env python3
"""
All plotting for MODEL_PIPELINE, in one module.

Sections
--------
  style            shared colours and axis styling, applied per figure (no
                   global matplotlib state is changed, so importing this module
                   is side-effect free apart from selecting the Agg backend)
  training curves  draw_curves(): training vs validation loss, the validation
                   loss split into its four terms (one run) or the selection
                   score (several runs), validation ALL / BOUNDARY / CORRUPTED
                   errors against the noisy input, and the misindexing gate.
                   train.py calls it after every epoch; the dotted line marks
                   the best epoch, the one saved as best.pth.

Add new figures as new sections with their own draw_* function and a
sub-command in main().

Usage
-----
    # redraw one run's curves from its history.json (default output: next to it)
    python plotting.py curves ../checkpoints/seed1

    # several runs on one figure, e.g. the three seeds
    python plotting.py curves ../checkpoints/seed{1,2,3} --out ../figures/seeds.png
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, MaxNLocator, NullFormatter

# ---------------------------------------------------------------------------
# style
# ---------------------------------------------------------------------------

INK, INK_2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
STYLE = {"font.size": 9, "axes.titlesize": 10, "text.color": INK, "axes.labelcolor": INK_2,
         "xtick.color": INK_2, "ytick.color": INK_2, "axes.edgecolor": MUTED}
PLAIN = FuncFormatter(lambda v, _: f"{v:g}")         # 0.42, not 4.2x10^-1


def _style_axis(ax, title, xlabel=None):
    ax.set_title(title, color=INK, loc="left")
    if xlabel:
        ax.set_xlabel(xlabel)
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(frameon=False, fontsize=8)


def _log_if_positive(ax):
    """Log y-axis when the axis holds positive data; otherwise say nothing was recorded."""
    ys = np.concatenate([np.asarray(line.get_ydata(), float) for line in ax.get_lines()
                         if line.get_linestyle() != ":"] or [np.array([])])
    ys = ys[np.isfinite(ys) & (ys > 0)]
    if ys.size == 0:
        ax.text(0.5, 0.5, "not recorded for this run", transform=ax.transAxes,
                ha="center", va="center", color=MUTED)
        return
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(PLAIN)
    lo, hi = ax.get_ylim()
    # Minor-tick labels only on a narrow range, where they are the only labels.
    ax.yaxis.set_minor_formatter(PLAIN if hi / lo < 10 else NullFormatter())


# ---------------------------------------------------------------------------
# training curves
# ---------------------------------------------------------------------------


def _val(history, key):
    return np.array([h["val"].get(key, np.nan) for h in history], float)


def best_epoch(history):
    """The epoch with the lowest validation score (the one saved as best.pth)."""
    scored = [h for h in history if "score" in h["val"]]
    return min(scored, key=lambda h: h["val"]["score"])["epoch"] if scored else None


def draw_curves(runs, path):
    """runs: {name: history list from history.json}. Writes one PNG to `path`."""
    runs = {name: hist for name, hist in runs.items() if hist}
    if not runs:
        return
    single = len(runs) == 1

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(2, 3, figsize=(15, 8.2))
        ax = axes.ravel()
        for r, (name, hist) in enumerate(runs.items()):
            ep = [h["epoch"] for h in hist]
            colour = SERIES[r % len(SERIES)]
            tag = "" if single else f"{name} "

            # loss: training vs validation (same formula)
            ax[0].plot(ep, [h["train_loss"] for h in hist], color=SERIES[0] if single else colour,
                       lw=2, ls="-" if single else "--", label=f"{tag}training")
            ax[0].plot(ep, _val(hist, "loss"), color=SERIES[1] if single else colour,
                       lw=2, label=f"{tag}validation")

            # one run: the four loss terms; several runs: the selection score
            if single:
                for k, part in enumerate(("main", "gate", "refine", "replace")):
                    ax[1].plot(ep, [h["val"].get("loss_parts", {}).get(part, np.nan) for h in hist],
                               color=SERIES[k], lw=2, label=part)
            else:
                ax[1].plot(ep, _val(hist, "score"), color=colour, lw=2, label=name)

            for i, key in ((2, "all"), (3, "bnd"), (4, "bad")):
                ax[i].plot(ep, _val(hist, key), color=colour, lw=2, label="model" if single else name)

            ax[5].plot(ep, _val(hist, "gate_precision"), color=SERIES[0] if single else colour,
                       lw=2, label=f"{tag}precision")
            ax[5].plot(ep, _val(hist, "gate_recall"), color=SERIES[1] if single else colour,
                       lw=2, ls="-" if single else "--", label=f"{tag}recall")

            best = best_epoch(hist)
            if best is not None:
                for a in ax:
                    a.axvline(best, color=MUTED if single else colour, ls=":", lw=1.2)

        # noisy-input reference: one line when every run shares the validation set
        inputs = {name: h[0]["val"]["in_all"] for name, h in runs.items() if "in_all" in h[0]["val"]}
        if inputs and len({round(v, 9) for v in inputs.values()}) == 1:
            ax[2].axhline(next(iter(inputs.values())), color=MUTED, ls="--", lw=1.2, label="noisy input")
        else:
            for name, v in inputs.items():
                ax[2].axhline(v, color=SERIES[list(runs).index(name) % len(SERIES)], ls="--", lw=1,
                              label=f"{name} input")

        # runs before 2026-10-13 recorded medians and no "stat" key
        stats = {h[0]["val"].get("stat", "median") for h in runs.values()}
        stat = stats.pop() if len(stats) == 1 else "mean or median"
        titles = ("loss (training vs validation)",
                  "validation loss components" if single else "validation score (lower = better)",
                  f"validation ALL, {stat} error (deg)", f"validation BOUNDARY, {stat} error (deg)",
                  f"validation CORRUPTED, {stat} error (deg)", "misindexing gate")
        last_epoch = max(h[-1]["epoch"] for h in runs.values())
        for i, (a, title) in enumerate(zip(ax, titles)):
            a.set_xlim(0.5, last_epoch + 0.5)
            a.xaxis.set_major_locator(MaxNLocator(integer=True))
            if i < 5:
                _log_if_positive(a)
            _style_axis(a, title, xlabel="epoch")
        ax[5].set_ylim(0, 1.02)

        if single:
            name, hist = next(iter(runs.items()))
            best = best_epoch(hist)
            fig.suptitle(name + (f"   (best epoch {best}, dotted line)" if best else ""),
                         x=0.01, ha="left", color=INK, fontsize=11)
        fig.tight_layout()
        fig.savefig(path, dpi=110)
        plt.close(fig)


def _load_runs(paths):
    """{run name: history} from run folders or history.json files."""
    runs, folders = {}, []
    for p in map(Path, paths):
        p = p / "history.json" if p.is_dir() else p
        runs[p.parent.name] = json.loads(p.read_text())
        folders.append(p.parent)
    return runs, folders


# ---------------------------------------------------------------------------
# command line
# ---------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser(description="MODEL_PIPELINE figures")
    sub = p.add_subparsers(dest="figure", required=True)
    c = sub.add_parser("curves", help="training curves of one or several runs")
    c.add_argument("runs", nargs="+", help="run folders (holding history.json) or history.json files")
    c.add_argument("--out", default=None, help="output PNG; default <run>/training_curves.png for "
                                               "one run, ./training_comparison.png for several")
    args = p.parse_args()

    if args.figure == "curves":
        runs, folders = _load_runs(args.runs)
        out = Path(args.out) if args.out else (
            folders[0] / "training_curves.png" if len(runs) == 1 else Path("training_comparison.png"))
        draw_curves(runs, out)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
