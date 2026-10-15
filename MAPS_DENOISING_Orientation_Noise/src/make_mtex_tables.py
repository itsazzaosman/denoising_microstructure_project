#!/usr/bin/env python3
"""
LaTeX tables of the network vs MTEX comparison, from the files that
compare_with_mtex.py writes (run that first).

  paper/table_mtex_main.tex   main text: seven representative test sets; for each,
                              the noisy input, the best MTEX filter of the set and
                              the network, on the four measures, mean (sd) over
                              the maps, with the Holm-corrected significance of
                              the network against that filter and the 95%
                              confidence interval of the difference in ALL.
  paper/table_mtex_full.tex   appendix: every test set and every filter, one table
                              per measure (means over the maps), with the
                              difference to the best filter, its 95% interval and
                              the Holm-corrected p-value.

Usage
-----
    python make_mtex_tables.py
"""

import csv
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
IN = ROOT / "figures/mtex_comparison"
FILTERS = ["clean-mean", "clean-median", "clean-kuwahara", "clean-spline",
           "clean-halfquad", "clean-infconv"]
NAMES = {"clean-mean": "Mean", "clean-median": "Median", "clean-kuwahara": "Kuwahara",
         "clean-spline": "Spline", "clean-halfquad": "Half-quadratic",
         "clean-infconv": "Infimal conv."}
SHORT = {"clean-mean": "Mean", "clean-median": "Median", "clean-kuwahara": "Kuwah.",
         "clean-spline": "Spline", "clean-halfquad": "Half-q.", "clean-infconv": "Inf.\\ conv."}
MEASURES = [("all", "all_mean", "ALL ($^\\circ$)"), ("bnd", "bnd_mean", "BOUNDARY ($^\\circ$)"),
            ("bad", "bad_mean", "CORRUPTED ($^\\circ$)"), ("gross", "gross_pct", "${>}10^\\circ$ (\\%)")]
MAIN_SETS = ["scatter1", "scatter5", "scatter15", "scatter1_mis9", "scatter5_mis9",
             "scatter15_mis9", "test_random"]


def label(s):
    m = re.match(r"scatter(\d+)(?:_mis(\d+))?$", s)
    if not m:
        return "mixed"
    return f"${m.group(1)}^\\circ$, ${m.group(2) or 0}\\%$"


def stars(p):
    if not np.isfinite(p) or p >= 0.05:
        return ""
    return "$^{***}$" if p < 0.001 else "$^{**}$" if p < 0.01 else "$^{*}$"


def load():
    vals = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))   # set, method, measure
    with open(IN / "per_map.csv") as f:
        for r in csv.DictReader(f):
            for key, col, _ in MEASURES:
                vals[r["set"]][r["method"]][key].append(float(r[col]))
    tests = {}
    with open(IN / "per_set_tests.csv") as f:
        for r in csv.DictReader(f):
            tests[r["set"], r["measure"]] = {k: (float(v) if k not in ("set", "measure", "best_mtex")
                                                 else v) for k, v in r.items()}
    sets = (ROOT / "configs/test_grid.txt").read_text().split() + ["test_random"]
    return vals, tests, [s for s in sets if s in vals]


def fmt(v, key):
    return f"{v:.2f}" if key == "gross" else f"{v:.3f}"


def main_table(vals, tests, out):
    lines = [
        "% Made by src/make_mtex_tables.py from figures/mtex_comparison/; do not edit by hand.",
        "\\begin{table*}[t]",
        "  \\centering",
        "  \\caption{The network against the best MTEX filter of each test set, on test maps",
        "    1--10. Seven of the 29 sets are shown; all are in Appendix~\\ref{app:mtex}.",
        "    Each value is the mean (standard deviation) over the 10 maps; for the",
        "    network, each map's value is the mean of the three training runs. The best",
        "    MTEX filter is the one with the lowest ALL error on the set, chosen after",
        "    scoring, and is used for all four measures. All MTEX filters were run with",
        "    pre-cleaning. The lower of the two methods is in bold. Stars: paired $t$-test",
        "    of the network against that filter over the 10 maps, Holm-corrected over the",
        "    29 sets: $^{*}p<0.05$, $^{**}p<0.01$, $^{***}p<0.001$; no star: not",
        "    significant. Last column: difference in ALL, network minus filter, with its",
        "    95\\% confidence interval.}",
        "  \\label{tab:mtex}",
        "  \\footnotesize",
        "  \\setlength{\\tabcolsep}{4pt}",
        "  \\begin{tabular}{@{}llrrrrr@{}}",
        "    \\toprule",
        "    $\\theta_{\\max}$, $p$ & Method & " + " & ".join(h for _, _, h in MEASURES)
        + " & $\\Delta$ALL [95\\% CI] \\\\",
        "    \\midrule",
    ]
    for i, s in enumerate(MAIN_SETS):
        best = tests[s, "all"]["best_mtex"]
        rows = {"input": "Noisy input", best: f"{NAMES[best]} (MTEX)", "ours": "Ours"}
        cells = {m: {} for m in rows}
        for key, _, _ in MEASURES:
            means = {m: np.mean(vals[s][m][key]) for m in rows}
            if not np.isfinite(means["ours"]):
                for m in rows:
                    cells[m][key] = "--"
                continue
            shown = {m: fmt(means[m], key) for m in rows}
            low = min(float(shown[best]), float(shown["ours"]))
            for m in rows:
                v = vals[s][m][key]
                txt = f"{shown[m]} ({fmt(np.std(v, ddof=1), key)})"
                if m != "input" and float(shown[m]) == low:
                    txt = f"\\textbf{{{shown[m]}}} ({fmt(np.std(v, ddof=1), key)})"
                if m == "ours":
                    txt += stars(tests[s, key]["p_holm"])
                cells[m][key] = txt
        t = tests[s, "all"]
        ci = f"${t['difference']:+.3f}$ [${t['ci_low']:+.3f}$, ${t['ci_high']:+.3f}$]"
        for j, m in enumerate(rows):
            first = label(s) if j == 0 else ""
            last = ci if m == "ours" else ""
            lines.append(f"    {first} & {rows[m]} & " + " & ".join(cells[m][k] for k, _, _ in MEASURES)
                         + f" & {last} \\\\")
        if i < len(MAIN_SETS) - 1:
            lines.append("    \\addlinespace")
    lines += ["    \\bottomrule", "  \\end{tabular}", "\\end{table*}", ""]
    out.write_text("\n".join(lines))


def full_tables(vals, tests, sets, out):
    lines = ["% Made by src/make_mtex_tables.py from figures/mtex_comparison/; do not edit by hand.",
             "\\section{Full comparison with the MTEX filters}",
             "\\label{app:mtex}",
             "",
             "Tables~\\ref{tab:mtex_full_all}--\\ref{tab:mtex_full_gross} give every test set",
             "and every MTEX filter (all with pre-cleaning), on test maps 1--10. Each value is",
             "the mean over the 10 maps; the standard deviations are in",
             "\\texttt{figures/mtex\\_comparison/summary.csv} and \\texttt{per\\_map.csv}. The",
             "lowest value of each row, the noisy input aside, is in bold. $\\Delta$ is the",
             "network minus the best filter of the set (the one with the lowest ALL error),",
             "with its 95\\% confidence interval, and $p$ the paired $t$-test over the 10",
             "maps, Holm-corrected over the sets.",
             ""]
    methods = ["input"] + FILTERS + ["ours"]
    for key, _, head in MEASURES:
        rows = [s for s in sets if np.isfinite(np.mean(vals[s]["ours"][key]))]
        lines += [
            "\\begin{table*}[p]",
            "  \\centering",
            f"  \\caption{{{head.split(' (')[0]}: network against every MTEX filter, all test sets "
            f"with {'misindexing' if key == 'bad' else 'their maps 1--10'}.}}",
            f"  \\label{{tab:mtex_full_{key}}}",
            "  \\footnotesize",
            "  \\setlength{\\tabcolsep}{3.5pt}",
            "  \\begin{tabular}{@{}l" + "r" * len(methods) + "rr@{}}",
            "    \\toprule",
            "    $\\theta_{\\max}$, $p$ & Input & " + " & ".join(SHORT[f] for f in FILTERS)
            + " & Ours & $\\Delta$ [95\\% CI] & $p$ \\\\",
            "    \\midrule",
        ]
        for s in rows:
            shown = {m: fmt(np.mean(vals[s][m][key]), key) for m in methods}
            low = min(float(shown[m]) for m in methods if m != "input")
            cells = [f"\\textbf{{{shown[m]}}}" if m != "input" and float(shown[m]) == low else shown[m]
                     for m in methods]
            t = tests[s, key]
            nd = "{:+.2f}" if key == "gross" else "{:+.3f}"
            ci = (f"${nd.format(t['difference'])}$ "
                  f"[${nd.format(t['ci_low'])}$, ${nd.format(t['ci_high'])}$]")
            p = t["p_holm"]
            ptxt = ("--" if not np.isfinite(p) else "$<$0.001" if p < 0.001
                    else f"{p:.3f}" if p < 0.01 else f"{p:.2f}")
            lines.append(f"    {label(s)} & " + " & ".join(cells) + f" & {ci} & {ptxt} \\\\")
        lines += ["    \\bottomrule", "  \\end{tabular}", "\\end{table*}", ""]
    out.write_text("\n".join(lines))


def main():
    vals, tests, sets = load()
    main_table(vals, tests, ROOT / "paper/table_mtex_main.tex")
    full_tables(vals, tests, sets, ROOT / "paper/table_mtex_full.tex")
    print(f"wrote {ROOT / 'paper/table_mtex_main.tex'} and {ROOT / 'paper/table_mtex_full.tex'}")


if __name__ == "__main__":
    main()
