"""Cross-detector figures from the score tables (AGENTS.md "Figures" rule).

Reads every results/<detector>/scores.csv (written by shared/score.py) and draws
three charts from the "all edited pages" rows. Numbers only: no page content.

    fig_auc.png                  (a) AUC with 95% range per detector and input size
    fig_ruleB_recall_vs_fa.png   (b) rule B recall against false-alarm rate, with ranges
    fig_location.png             (c) location recall (with range) next to false regions (count)

Each figure is saved as PNG at 300 dpi and as PDF (for Overleaf). It works with
whatever detectors have a scores.csv, so DTD and DAF appear when their results exist.

Run from the repo root:
    python3 analysis/figures/make_figures.py
    python3 analysis/figures/make_figures.py --out-dir results/combined/figures   # to commit the PNGs
By default the figures go to analysis/figures/out/, which is not committed.
"""

import argparse
import csv
from pathlib import Path

import matplotlib

# Draw to files only (no window), so it also runs without a screen.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Repo root, found from this file's place: <root>/analysis/figures/make_figures.py.
REPO = Path(__file__).resolve().parents[2]

# Input sizes in the order they are drawn; any other size found is added after these.
SIZE_ORDER = ["shrink512", "full"]
# One marker shape and one colour per detector (shape alone must tell them apart in greyscale).
MARKERS = ["o", "s", "^", "D", "v", "P", "X"]
COLOURS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000"]
# Input size is told apart by fill, error-bar line style and hatch, not by colour.
SIZE_STYLE = {
    "shrink512": {"filled": False, "line": "--", "hatch": "///"},
    "full": {"filled": True, "line": "-", "hatch": ""},
}


def load(results_root):
    """Read results/*/scores.csv: returns {(detector, size): {metric: (value, low, high), 'n': (...)}}.

    Only the 'all edited pages' rows (breakdown all, subset all) are used.
    """
    data = {}
    for path in sorted(Path(results_root).glob("*/scores.csv")):
        with open(path, encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if (row["breakdown"], row["subset"]) != ("all", "all"):
                    continue
                entry = data.setdefault((row["detector"], row["size"]), {})
                # An empty cell (no range could be computed) becomes NaN.
                num = lambda text: float(text) if text != "" else float("nan")
                entry[row["metric"]] = (num(row["value"]), num(row["ci_low"]), num(row["ci_high"]))
                entry["n"] = (int(row["n_tampered"]), int(row["n_untouched"]), int(row["n_edits"]),
                              int(row["n_found"]), int(row["n_false_regions"]))
    if not data:
        raise SystemExit(f"no scores.csv found under {results_root}")
    return data


def orders(data):
    """Detectors (sorted) and the sizes present (known order first)."""
    detectors = sorted({d for d, _ in data})
    present = {s for _, s in data}
    sizes = [s for s in SIZE_ORDER if s in present] + sorted(present - set(SIZE_ORDER))
    return detectors, sizes


def n_label(data, detector, sizes):
    """Second tick line: number of edited and untouched pages (flagged if it differs by size)."""
    counts = {data[(detector, s)]["n"][:2] for s in sizes if (detector, s) in data}
    first = sorted(counts)[0]
    text = f"n = {first[0]:,} edited / {first[1]:,} untouched"
    return text + " (varies by size)" if len(counts) > 1 else text


def err(value, low, high):
    """Error-bar lengths from a value and its 95% range (0 if the range is missing)."""
    if np.isnan(low) or np.isnan(high):
        return 0.0, 0.0
    return max(value - low, 0.0), max(high - value, 0.0)


def style_axes(ax):
    # Light horizontal grid, no top/right box lines.
    ax.grid(axis="y", color="#cccccc", linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def save(fig, out_dir, name):
    # Save the PNG (300 dpi) and the PDF (for Overleaf); return the PNG path.
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    return out_dir / f"{name}.png"


def fig_auc(data, out_dir):
    """(a) AUC with 95% range per detector and size, with a line at 0.5."""
    detectors, sizes = orders(data)
    fig, ax = plt.subplots(figsize=(max(4.8, 2.2 * len(detectors) + 2.6), 4.4))
    offsets = np.linspace(-0.16, 0.16, len(sizes)) if len(sizes) > 1 else [0.0]
    for i, detector in enumerate(detectors):
        colour = COLOURS[i % len(COLOURS)]
        for off, size in zip(offsets, sizes):
            if (detector, size) not in data:
                continue
            value, low, high = data[(detector, size)]["auc"]
            style = SIZE_STYLE.get(size, SIZE_STYLE["full"])
            lo_err, hi_err = err(value, low, high)
            ax.errorbar(i + off, value, yerr=[[lo_err], [hi_err]], fmt=MARKERS[i % len(MARKERS)], color=colour,
                        markerfacecolor=colour if style["filled"] else "white", markersize=8, markeredgewidth=1.6,
                        elinewidth=1.6, capsize=4, linestyle="none", label=f"{detector}, {size}")
    # Chance level.
    ax.axhline(0.5, color="black", linestyle=":", linewidth=1.2, label="chance (AUC = 0.5)")
    ax.set_xticks(range(len(detectors)))
    ax.set_xticklabels([f"{d}\n{n_label(data, d, sizes)}" for d in detectors], fontsize=8)
    ax.set_xlim(-0.6, len(detectors) - 0.4)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("AUC (area under ROC curve, 0 to 1)")
    ax.set_title("AUC per detector and input size (95% range)", fontsize=11)
    style_axes(ax)
    ax.legend(fontsize=8, loc="upper left", framealpha=0.95)
    return save(fig, out_dir, "fig_auc")


def fig_rule_b(data, out_dir):
    """(b) Rule B recall against false-alarm rate, with ranges on both axes."""
    detectors, sizes = orders(data)
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    for i, detector in enumerate(detectors):
        colour = COLOURS[i % len(COLOURS)]
        for size in sizes:
            if (detector, size) not in data:
                continue
            d = data[(detector, size)]
            fa, fa_lo, fa_hi = d["false_alarm_B"]
            rec, rec_lo, rec_hi = d["recall_B"]
            style = SIZE_STYLE.get(size, SIZE_STYLE["full"])
            xe, ye = err(fa, fa_lo, fa_hi), err(rec, rec_lo, rec_hi)
            n = d["n"]
            ax.errorbar(fa, rec, xerr=[[xe[0]], [xe[1]]], yerr=[[ye[0]], [ye[1]]], fmt=MARKERS[i % len(MARKERS)],
                        color=colour, markerfacecolor=colour if style["filled"] else "white", markersize=8,
                        markeredgewidth=1.6, elinewidth=1.4, capsize=3, linestyle="none",
                        label=f"{detector}, {size} (n = {n[0]:,} / {n[1]:,})")
    # A detector with no skill lies on this line (recall equals false-alarm rate).
    ax.plot([0, 1], [0, 1], color="black", linestyle=":", linewidth=1.2, label="no skill (recall = false-alarm rate)")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("False-alarm rate on untouched pages (rule B, 0 to 1)")
    ax.set_ylabel("Recall on edited pages (rule B, 0 to 1)")
    ax.set_title("Rule B: recall against false alarms (95% ranges)", fontsize=11)
    ax.grid(color="#cccccc", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(fontsize=7.5, loc="lower right", framealpha=0.95, title="n = edited / untouched pages", title_fontsize=7.5)
    return save(fig, out_dir, "fig_ruleB_recall_vs_fa")


def fig_location(data, out_dir):
    """(c) Location recall (left, with range) next to false regions (right, count, log scale)."""
    detectors, sizes = orders(data)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(8.4, 2.6 * len(detectors) + 5.0), 4.6))
    width = 0.8 / len(sizes)
    top_count = 1
    for i, detector in enumerate(detectors):
        colour = COLOURS[i % len(COLOURS)]
        for j, size in enumerate(sizes):
            if (detector, size) not in data:
                continue
            d = data[(detector, size)]
            style = SIZE_STYLE.get(size, SIZE_STYLE["full"])
            x = i + (j - (len(sizes) - 1) / 2) * width
            value, low, high = d["loc_recall"]
            lo_err, hi_err = err(value, low, high)
            # Left: recall as a bar with its 95% range.
            ax1.bar(x, value, width * 0.9, color=colour if style["filled"] else "white", edgecolor=colour,
                    hatch=style["hatch"], linewidth=1.5, label=f"{detector}, {size}")
            ax1.errorbar(x, value, yerr=[[lo_err], [hi_err]], fmt="none", ecolor="black", elinewidth=1.4, capsize=3)
            # Right: false regions, a plain count (the scorer gives no range for counts).
            ax2.bar(x, d["n"][4], width * 0.9, color=colour if style["filled"] else "white", edgecolor=colour,
                    hatch=style["hatch"], linewidth=1.5)
            # Print the count on the bar so the exact number can be read.
            ax2.annotate(f"{d['n'][4]:,}", (x, d["n"][4]), ha="center", va="bottom", fontsize=8,
                         xytext=(0, 2), textcoords="offset points")
            top_count = max(top_count, d["n"][4])
    for ax in (ax1, ax2):
        ax.set_xticks(range(len(detectors)))
        ax.set_xticklabels([f"{d}\n{n_label(data, d, sizes)}" for d in detectors], fontsize=8)
        style_axes(ax)
    ax1.set_ylim(0, 1.0)
    ax1.set_ylabel("Location recall: edits found / all edits (0 to 1)")
    ax1.set_title("Edits found (95% range)", fontsize=11)
    ax1.legend(fontsize=8, loc="upper left", framealpha=0.95)
    ax2.set_yscale("log")
    # Start the log axis at 1 so bar heights are honest; leave room above for the count labels.
    ax2.set_ylim(1, top_count * 4)
    ax2.set_ylabel("False regions (count, log scale)")
    ax2.set_title("False regions on all pages", fontsize=11)
    fig.suptitle("Location level: edits found next to false regions", fontsize=12, y=1.02)
    fig.tight_layout()
    return save(fig, out_dir, "fig_location")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-root", type=Path, default=REPO / "results")
    parser.add_argument("--out-dir", type=Path, default=REPO / "analysis" / "figures" / "out")
    args = parser.parse_args()
    data = load(args.results_root)
    detectors, sizes = orders(data)
    print(f"detectors: {detectors}; sizes: {sizes}")
    for path in (fig_auc(data, args.out_dir), fig_rule_b(data, args.out_dir), fig_location(data, args.out_dir)):
        print("wrote", path.relative_to(REPO) if REPO in path.resolve().parents else path, "(+ .pdf)")


if __name__ == "__main__":
    main()
