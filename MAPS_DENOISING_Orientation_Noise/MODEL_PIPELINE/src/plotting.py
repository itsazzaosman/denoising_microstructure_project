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
                   draw_zoom(): the zoomed version of two of those panels,
                   training vs validation loss and the gate's precision / recall
                   in %, optionally for the first N epochs only.
  test curves      score_test_grid() + draw_test_curves(): the model against the
                   six MTEX filters on the 28 fixed-level test sets, one figure
                   per measure (ALL, BOUNDARY, CORRUPTED, >10 deg). Scored with
                   score_dataset.py's own functions on the maps every method has
                   (MTEX: maps 1-10), so the numbers are those of `make score-grid`.
  maps             load_map_methods() + draw_filter_grid(): one test map cleaned
                   by every method (clean, noisy, the six MTEX filters, the model)
                   as a 3 x 3 grid of IPF-Z maps (after Atindama et al. 2023, Fig. 2).
                   Replaces SHARED/src/plot_maps.py and plot_all_filters.py.

Add new figures as new sections with their own draw_* function and a
sub-command in main().

Usage
-----
    # redraw one run's curves from its history.json (default output: next to it)
    python plotting.py curves ../checkpoints/seed1

    # several runs on one figure, e.g. the three seeds
    python plotting.py curves ../checkpoints/seed{1,2,3} --out ../figures/seeds.png

    # zoomed loss + gate figure (default output: MODEL_PIPELINE/figures/seed1/training_zoom.png);
    # --epochs 19 shows epochs 1-19 only (-> training_zoom_epoch19.png)
    python plotting.py zoom ../checkpoints/seed1 --epochs 19

    # model vs MTEX on the test grid, from datasets/results_test_set_model/seed1
    # (default output: MODEL_PIPELINE/figures/seed1/curves_*.png)
    python plotting.py test --seed seed1

    # one map, every method, as the 3 x 3 grid (MTEX has maps 1-10 only;
    # default output: MODEL_PIPELINE/figures/seed1/maps/<set>_map<map>_grid.png)
    python plotting.py maps --seed seed1 --set scatter5_mis3 --map 2
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import (FixedLocator, FuncFormatter, LogLocator, MaxNLocator,
                               NullFormatter, NullLocator)

# ---------------------------------------------------------------------------
# style
# ---------------------------------------------------------------------------

INK, INK_2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
STYLE = {"font.size": 9, "axes.titlesize": 10, "text.color": INK, "axes.labelcolor": INK_2,
         "xtick.color": INK_2, "ytick.color": INK_2, "axes.edgecolor": MUTED}
PLAIN = FuncFormatter(lambda v, _: f"{v:g}")         # 0.42, not 4.2x10^-1
FIGURES = Path(__file__).resolve().parents[1] / "figures"           # MODEL_PIPELINE/figures
SHARED_SRC = Path(__file__).resolve().parents[2] / "SHARED" / "src"
MAP_SHAPE = (128, 128)


def _use_shared():
    """Make SHARED/src (orientation maths, paths, scoring) importable."""
    if str(SHARED_SRC) not in sys.path:
        sys.path.insert(0, str(SHARED_SRC))


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


def draw_zoom(name, history, path, epochs=None):
    """Two panels of one run, larger than in draw_curves: training vs validation loss
    (log scale) and the misindexing gate's precision and recall on the validation set,
    in %, on an axis zoomed to the data. epochs = N shows epochs 1-N only; the dotted
    line is the best epoch among those shown."""
    total = history[-1]["epoch"]
    hist = [h for h in history if h["epoch"] <= epochs] if epochs else history
    if not hist:
        raise SystemExit(f"no epochs to show (the run has {total})")
    ep = [h["epoch"] for h in hist]
    best = best_epoch(hist)
    with plt.rc_context(STYLE):
        fig, (ax_loss, ax_gate) = plt.subplots(1, 2, figsize=(14.4, 5.2))
        ax_loss.plot(ep, [h["train_loss"] for h in hist], color=SERIES[0], lw=2, marker="o",
                     ms=3.5, label="training")
        ax_loss.plot(ep, _val(hist, "loss"), color=SERIES[1], lw=2, marker="o", ms=3.5,
                     label="validation")
        _log_if_positive(ax_loss)
        ax_loss.set_ylabel("loss (log scale)")
        ax_gate.plot(ep, 100 * _val(hist, "gate_precision"), color=SERIES[0], lw=2, marker="o",
                     ms=3.5, label="precision")
        ax_gate.plot(ep, 100 * _val(hist, "gate_recall"), color=SERIES[1], lw=2, marker="o",
                     ms=3.5, label="recall")
        ax_gate.set_ylabel("%")
        for ax, title in ((ax_loss, "Training vs validation loss"),
                          (ax_gate, "Misindexing detection (validation)")):
            ax.set_xlim(0.5, ep[-1] + 0.5)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            if best is not None:
                ax.axvline(best, color=MUTED, ls=":", lw=1.2)
            _style_axis(ax, title, xlabel="epoch")
        if best is not None:
            ax_gate.text(best, ax_gate.get_ylim()[0], f" best epoch {best}", color=MUTED,
                         fontsize=8, va="bottom")
        fig.suptitle(f"{name}, epochs 1-{ep[-1]} of {total}", x=0.01, ha="left", color=INK,
                     fontsize=11)
        fig.tight_layout()
        fig.savefig(path, dpi=100)
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
# test curves: the model against the MTEX filters on the test grid
# ---------------------------------------------------------------------------

SCATTER = (1, 3, 5, 7, 9, 12, 15)          # maximum scatter angles of the test grid, deg
MISINDEX = (0, 3, 6, 9)                    # misindexed pixels of the test grid, %
FILTERS = {"clean-mean": "mean", "clean-median": "median", "clean-kuwahara": "Kuwahara",
           "clean-spline": "spline", "clean-halfquad": "half-quadratic",
           "clean-infconv": "infimal conv."}
NOISY = "noisy (no filter)"
# key: (file name, y-axis label, misindexing levels drawn, legend place)
MEASURES = {
    "all":   ("all", "mean error, all pixels (deg)", MISINDEX, "lower right"),
    "bnd":   ("boundary", "mean error, boundary pixels (deg)", MISINDEX, "lower right"),
    "bad":   ("corrupted", "mean error, misindexed pixels (deg)", MISINDEX[1:], "lower right"),
    "gross": ("gross", "pixels more than 10° off (% of map)", MISINDEX, "upper left"),
}


def _set_name(scatter, mis):
    return f"scatter{scatter}" + (f"_mis{mis}" if mis else "")


def score_test_grid(seed, sets=None, shape=MAP_SHAPE):
    """Per-map errors of every method on every test set, using only the maps that
    every method has. Returns ({set: {method: {measure: per-map values}}}, model names)."""
    _use_shared()
    from paths import CLEAN_DIR, MODEL_RESULTS, MTEX_RESULTS, TEST_SETS
    from score_dataset import index_dataset, score_one

    scores, models = {}, set()
    for name in sets or [_set_name(s, m) for m in MISINDEX for s in SCATTER]:
        model_dir = MODEL_RESULTS / seed / name
        if not model_dir.is_dir():
            print(f"  {name:<15} skipped: no model results in {model_dir}")
            continue
        jobs = index_dataset(CLEAN_DIR, TEST_SETS / f"noise_{name}", [model_dir, MTEX_RESULTS / name])
        methods = set().union(*(j["results"] for j in jobs))
        jobs = [j for j in jobs if len(j["results"]) == len(methods)]
        per_map = [s for s in (score_one(j, shape) for j in jobs) if s]
        scores[name] = {m: {k: np.array([s[m][k] for s in per_map if m in s]) for k in MEASURES}
                        for m in [NOISY, *methods]}
        models |= {f.stem.split("__")[-1] for f in model_dir.glob("*__*.txt")}
        print(f"  {name:<15} {len(per_map)} maps x {len(methods)} methods", flush=True)
    return scores, sorted(models)


def _mean_sd(v):
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.nan, np.nan
    return float(v.mean()), float(v.std(ddof=1)) if v.size > 1 else 0.0


def _style_test_axis(ax, gross):
    if gross:   # zeros are common here, so the axis is linear below 0.01 % and log above
        ax.set_yscale("symlog", linthresh=0.01, linscale=0.4)
        ax.yaxis.set_major_locator(FixedLocator([0, 0.01, 0.1, 1, 10]))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.set_ylim(-0.002, 70)
    else:
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(subs=(1, 2, 5)))
        ax.yaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(PLAIN)
    ax.set_xticks(SCATTER)
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def draw_test_curves(scores, models, out_dir):
    """One figure per measure: error against the maximum scatter angle, one panel per
    misindexing level, one line per method. Points are the mean over the maps, bars the
    standard deviation across them (the spread between maps, not a confidence interval).
    Returns the paths written."""
    out_dir.mkdir(parents=True, exist_ok=True)
    x = np.array(SCATTER, float)
    lines = [(m, label, SERIES[i + 1], False) for i, (m, label) in enumerate(FILTERS.items())]
    lines += [(m, m, SERIES[0] if i == 0 else SERIES[7], True) for i, m in enumerate(models)]
    written = []
    for key, (stem, ylabel, levels, legend) in MEASURES.items():
        gross, square = key == "gross", len(levels) == 4
        with plt.rc_context(STYLE):
            fig, axes = plt.subplots(2 if square else 1, 2 if square else 3, sharex=True,
                                     sharey=True, squeeze=False,
                                     figsize=(8.4, 6.6) if square else (11.0, 3.8))
            for ax, mis, letter in zip(axes.flat, levels, "abcd"):
                sets = [_set_name(s, mis) for s in SCATTER]
                stat = lambda m: np.array([_mean_sd(scores[s][m][key])
                                           if m in scores.get(s, {}) else (np.nan, np.nan)
                                           for s in sets]).T
                ax.plot(x, stat(NOISY)[0], color=MUTED, ls="--", lw=1.2, label="noisy input")
                for m, label, colour, ours in lines:
                    mu, sd = stat(m)
                    low = np.minimum(sd, mu if gross else 0.9 * mu)   # bar stays above the axis floor
                    ax.errorbar(x, mu, yerr=[low, sd], color=colour, lw=2.6 if ours else 1.1,
                                marker="o", ms=4.5 if ours else 2.8, elinewidth=0.8, capsize=1.5,
                                zorder=3 if ours else 2, label=label)
                _style_test_axis(ax, gross)
                ax.set_title(f"({letter}) misindexed pixels $p$ = {mis}%", loc="left", color=INK)
            for ax in axes[-1]:
                ax.set_xlabel(r"maximum scatter angle $\theta_{\max}$ (deg)")
            for ax in axes[:, 0]:
                ax.set_ylabel(ylabel)
            axes[0, 0].legend(frameon=False, fontsize=7.5, ncol=2, loc=legend)
            fig.tight_layout()
            path = out_dir / f"curves_{stem}.png"
            fig.savefig(path, dpi=200)
            plt.close(fig)
            written.append(path)
    return written


# ---------------------------------------------------------------------------
# maps: one test map, cleaned by every method
# ---------------------------------------------------------------------------


def _set_label(name):
    scatter, _, mis = name.removeprefix("scatter").partition("_mis")
    return rf"$\theta_{{\max}}$ = {scatter}°, $p$ = {mis or 0}%"


def load_map_methods(seed, set_name, map_id):
    """The clean map and every method's version of one test map. Returns (q_clean,
    [(name, q, is_model)]): the noisy input, the MTEX filters MTEX has for this map,
    then the model(s) found in datasets/results_test_set_model/<seed>/<set>."""
    _use_shared()
    from orientation import euler_to_quat, load_euler
    from paths import MODEL_RESULTS, MTEX_RESULTS, clean_map, noise_dir

    if not noise_dir(set_name).is_dir():
        sys.exit(f"no test set {set_name!r} (expected {noise_dir(set_name)})")
    stem = f"map_{map_id:05d}_clean_euler_noisy"
    read = lambda path: euler_to_quat(load_euler(path))
    q_clean = read(clean_map(map_id))
    panels = [("noisy input", read(noise_dir(set_name) / f"{stem}.txt"), False)]
    for f, name in FILTERS.items():
        path = MTEX_RESULTS / set_name / f"{stem}_noisy__{f}.txt"
        if path.exists():
            panels.append((f"MTEX {name}", read(path), False))
    models = sorted((MODEL_RESULTS / seed / set_name).glob(f"{stem}_noisy__*.txt"))
    if not models:
        print(f"note: no model result for map {map_id} in {MODEL_RESULTS / seed / set_name}")
    panels += [(path.stem.split("__")[-1], read(path), True) for path in models]
    return q_clean, panels


def draw_filter_grid(q_clean, panels, title, out_stem):
    """The paper figure: a 3 x 3 grid of IPF-Z maps, clean / noisy / the six MTEX filters /
    the model, each titled with its mean error, plus the IPF colour key. Needs all six MTEX
    filters, which exist for maps 1-10 only. Writes <out_stem>_grid.png; returns its path,
    or None when a method is missing."""
    _use_shared()
    from orientation import disorientation_deg, ipf_z_color
    from plot_noise_examples import ipf_key

    mtex = [p for p in panels if p[0].startswith("MTEX ")]
    models = [p for p in panels if p[2]]
    if len(mtex) != len(FILTERS) or not models:
        print(f"grid figure skipped: it needs all {len(FILTERS)} MTEX filters and the model "
              f"(found {len(mtex)} filters, {len(models)} model); MTEX has maps 1-10 only")
        return None
    maps = [("clean", q_clean, False), panels[0], *mtex, models[0]]
    with plt.rc_context({**STYLE, "font.size": 8}):
        fig = plt.figure(figsize=(7.0, 7.0))
        gs = fig.add_gridspec(3, 4, width_ratios=[1, 1, 1, 0.55], wspace=0.06, hspace=0.22,
                              left=0.01, right=0.99, top=0.9, bottom=0.01)
        for k, (name, q, ours) in enumerate(maps):
            ax = fig.add_subplot(gs[k // 3, k % 3])
            ax.imshow(ipf_z_color(q).reshape(*MAP_SHAPE, 3), interpolation="nearest")
            err = "" if k == 0 else f"\nmean error {disorientation_deg(q_clean, q).mean():.2f}°"
            ax.set_title(f"({'abcdefghi'[k]}) {name}{err}", fontsize=8.5, color=INK,
                         weight="bold" if ours else "normal")
            ax.axis("off")
        ipf_key(fig.add_subplot(gs[1, 3]))
        fig.suptitle(title, fontsize=10, color=INK)
        path = Path(f"{out_stem}_grid.png")
        fig.savefig(path, dpi=300)
        plt.close(fig)
    return path


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
    z = sub.add_parser("zoom", help="zoomed loss and gate panels of one run")
    z.add_argument("run", help="run folder (holding history.json) or a history.json file")
    z.add_argument("--epochs", type=int, default=None, help="show epochs 1-N only")
    z.add_argument("--out", default=None, help="output PNG; default MODEL_PIPELINE/figures/<run>/"
                                               "training_zoom[_epochN].png")
    t = sub.add_parser("test", help="model vs MTEX on the test grid (4 figures)")
    t.add_argument("--seed", default="seed1",
                   help="model results folder, datasets/results_test_set_model/<seed>")
    t.add_argument("--sets", nargs="+", default=None, help="only these test sets (default: all 28)")
    t.add_argument("--out", default=None, help="output folder; default MODEL_PIPELINE/figures/<seed>")
    m = sub.add_parser("maps", help="one test map cleaned by every method (3 x 3 grid)")
    m.add_argument("--seed", default="seed1",
                   help="model results folder, datasets/results_test_set_model/<seed>")
    m.add_argument("--set", default="scatter5_mis3", help="test set, e.g. scatter15_mis9")
    m.add_argument("--map", type=int, default=1, help="map number (MTEX has maps 1-10)")
    m.add_argument("--out", default=None,
                   help="output folder; default MODEL_PIPELINE/figures/<seed>/maps")
    args = p.parse_args()

    if args.figure == "curves":
        runs, folders = _load_runs(args.runs)
        out = Path(args.out) if args.out else (
            folders[0] / "training_curves.png" if len(runs) == 1 else Path("training_comparison.png"))
        draw_curves(runs, out)
        print(f"wrote {out}")
    elif args.figure == "zoom":
        runs, _ = _load_runs([args.run])
        name, history = next(iter(runs.items()))
        tag = f"_epoch{args.epochs}" if args.epochs else ""
        out = Path(args.out) if args.out else FIGURES / name / f"training_zoom{tag}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        draw_zoom(name, history, out, args.epochs)
        print(f"wrote {out}")
    elif args.figure == "test":
        scores, models = score_test_grid(args.seed, args.sets)
        out = Path(args.out) if args.out else FIGURES / args.seed
        for path in draw_test_curves(scores, models, out):
            print(f"wrote {path}")
    elif args.figure == "maps":
        q_clean, panels = load_map_methods(args.seed, args.set, args.map)
        out = Path(args.out) if args.out else FIGURES / args.seed / "maps"
        out.mkdir(parents=True, exist_ok=True)
        stem = out / f"{args.set}_map{args.map:03d}"
        title = f"Test map {args.map}, {_set_label(args.set)}"
        path = draw_filter_grid(q_clean, panels, title, stem)
        if path:
            print(f"wrote {path}")


if __name__ == "__main__":
    main()
