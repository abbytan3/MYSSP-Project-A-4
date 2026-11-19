"""T16: the five ADCD-Net analysis figures (numbers only, PNG only).

Reads results/adcd/scores.csv and results/adcd/differences.csv (already scored by
shared/score.py; this script never reruns the scorer and never reads .npz files).
Re-uses the style helpers from make_figures.py in this folder.

    fig1_auc.png            AUC at shrink512 vs full, with 95% ranges
    fig2_rule_b.png         rule B recall vs false-alarm rate at both sizes (rule A for contrast)
    fig3_location.png       location recall at both sizes, with the false-region count beside it
    fig4_ai_vs_hand.png     AI vs handcrafted: AUC and location recall at both sizes, plus the difference
    fig5_breakdowns.png     AUC by editor/method and by capture condition at both sizes, with n

Run from the repo root (needs matplotlib and numpy from analysis/figures/requirements.txt):
    python3 analysis/figures/make_figures_t16.py
"""

import argparse
import csv
import sys
from pathlib import Path

# Make the sibling script importable however this file is started.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Re-use the helpers and look of the existing figure script.
from make_figures import COLOURS, SIZE_STYLE, err, style_axes  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
SIZES = ["shrink512", "full"]
# One marker shape per group so greyscale prints still tell them apart.
GROUP_MARK = {"ai": "o", "handcrafted": "s"}
GROUP_NAME = {"ai": "AI-edited", "handcrafted": "Handcrafted"}
GROUP_COLOUR = {"ai": COLOURS[0], "handcrafted": COLOURS[1]}
GROUP_HATCH = {"ai": "", "handcrafted": "xx"}


def num(text):
    # An empty cell (no range computed) becomes NaN.
    return float(text) if text != "" else float("nan")


def load_scores(path):
    """scores.csv -> {(size, breakdown, subset, metric): row dict with value, low, high and counts}."""
    table = {}
    with open(path, encoding="utf-8", newline="") as handle:
        for r in csv.DictReader(handle):
            table[(r["size"], r["breakdown"], r["subset"], r["metric"])] = {
                "v": num(r["value"]), "lo": num(r["ci_low"]), "hi": num(r["ci_high"]),
                "n_t": int(r["n_tampered"]), "n_u": int(r["n_untouched"]), "n_e": int(r["n_edits"]),
                "n_f": int(r["n_found"]), "n_fr": int(r["n_false_regions"]),
            }
    return table


def load_diffs(path):
    """differences.csv -> {(comparison, size, subset, metric): (difference, low, high)}."""
    table = {}
    with open(path, encoding="utf-8", newline="") as handle:
        for r in csv.DictReader(handle):
            table[(r["comparison"], r["size"], r["subset"], r["metric"])] = (
                num(r["difference"]), num(r["ci_low"]), num(r["ci_high"]))
    return table


def save_png(fig, out_dir, name):
    # PNG only, 300 dpi (the task asks for PNG only).
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def errbars(ax, x, row, marker, colour, size, label=None, horizontal=False, ms=8):
    """One point with its 95% range, filled for full and hollow for shrink512."""
    style = SIZE_STYLE[size]
    lo, hi = err(row["v"], row["lo"], row["hi"])
    kw = dict(fmt=marker, color=colour, markerfacecolor=colour if style["filled"] else "white",
              markersize=ms, markeredgewidth=1.6, elinewidth=1.6, capsize=4, linestyle="none", label=label)
    if horizontal:
        ax.errorbar(row["v"], x, xerr=[[lo], [hi]], **kw)
    else:
        ax.errorbar(x, row["v"], yerr=[[lo], [hi]], **kw)


def fmt(row, digits=3):
    # "0.503 (0.486 to 0.522)"
    return f"{row['v']:.{digits}f} ({row['lo']:.{digits}f} to {row['hi']:.{digits}f})"


# --------------------------------------------------------------------------- figure 1
def fig1(s, out):
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    for i, size in enumerate(SIZES):
        row = s[(size, "all", "all", "auc")]
        errbars(ax, i, row, "o", "black", size, label=f"{size}: AUC {fmt(row)}", ms=9)
        ax.annotate(f"{row['v']:.3f}\n({row['lo']:.3f} to {row['hi']:.3f})", (i, row["hi"]), ha="center",
                    va="bottom", fontsize=8, xytext=(0, 5), textcoords="offset points")
    ax.axhline(0.5, color="black", linestyle=":", linewidth=1.2, label="chance (AUC = 0.5)")
    n = s[("full", "all", "all", "auc")]
    ax.set_xticks(range(2))
    ax.set_xticklabels(SIZES)
    ax.set_xlim(-0.6, 1.6)
    # Zoomed axis (0.4 to 0.7) so the gap to chance is visible; stated in the label.
    ax.set_ylim(0.4, 0.7)
    ax.set_ylabel("AUC (0 to 1; axis shows 0.4 to 0.7 only)")
    ax.set_xlabel("Input size")
    ax.set_title("ADCD-Net AUC by input size, all pages (95% range)", fontsize=11)
    ax.text(0.5, -0.2, f"n = {n['n_t']:,} edited pages and {n['n_u']:,} untouched pages at both sizes",
            transform=ax.transAxes, ha="center", fontsize=8)
    style_axes(ax)
    ax.legend(fontsize=8, loc="upper left", framealpha=0.95)
    return save_png(fig, out, "fig1_auc")


# --------------------------------------------------------------------------- figure 2
def fig2(s, out):
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    for size in SIZES:
        a = s[(size, "all", "all", "false_alarm_B")]
        r = s[(size, "all", "all", "recall_B")]
        xe, ye = err(a["v"], a["lo"], a["hi"]), err(r["v"], r["lo"], r["hi"])
        style = SIZE_STYLE[size]
        ax.errorbar(a["v"], r["v"], xerr=[[xe[0]], [xe[1]]], yerr=[[ye[0]], [ye[1]]], fmt="o", color="black",
                    markerfacecolor="black" if style["filled"] else "white", markersize=9, markeredgewidth=1.6,
                    elinewidth=1.4, capsize=3, linestyle="none",
                    label=f"RULE B (headline), {size}:\n   recall {fmt(r)}\n   false alarms {fmt(a)}")
    # Rule A for contrast with rule B at ONE size only (shrink512). It is not drawn at full size, where it
    # flags every page, and it is never used to compare sizes.
    for size in ["shrink512"]:
        a = s[(size, "all", "all", "false_alarm_A")]
        r = s[(size, "all", "all", "recall_A")]
        xe, ye = err(a["v"], a["lo"], a["hi"]), err(r["v"], r["lo"], r["hi"])
        style = SIZE_STYLE[size]
        ax.errorbar(a["v"], r["v"], xerr=[[xe[0]], [xe[1]]], yerr=[[ye[0]], [ye[1]]], fmt="^", color="#777777",
                    markerfacecolor="#777777" if style["filled"] else "white", markersize=8, markeredgewidth=1.4,
                    elinewidth=1.1, capsize=3, linestyle="none",
                    label="rule A, shrink512 only (contrast with rule B; not for comparing sizes)")
    ax.plot([0, 1], [0, 1], color="black", linestyle=":", linewidth=1.2, label="no skill (recall = false-alarm rate)")
    n = s[("full", "all", "all", "recall_B")]
    ax.set_xlim(0, 1.03)
    ax.set_ylim(0, 1.03)
    ax.set_xlabel("False-alarm rate on untouched pages (0 to 1)")
    ax.set_ylabel("Recall on edited pages (0 to 1)")
    ax.set_title("ADCD-Net rule B: recall vs false-alarm rate (95% ranges)\nrule A (shrink512 only) in grey for contrast",
                 fontsize=10.5)
    ax.text(0.5, -0.17, f"n = {n['n_t']:,} edited pages and {n['n_u']:,} untouched pages", transform=ax.transAxes,
            ha="center", fontsize=8)
    ax.grid(color="#cccccc", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(fontsize=6.8, loc="lower left", framealpha=0.95)
    return save_png(fig, out, "fig2_rule_b")


# --------------------------------------------------------------------------- figure 3
def fig3(s, out):
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    for i, size in enumerate(SIZES):
        row = s[(size, "all", "all", "loc_recall")]
        style = SIZE_STYLE[size]
        ax.bar(i, row["v"], 0.55, color="#555555" if style["filled"] else "white", edgecolor="black",
               hatch=style["hatch"], linewidth=1.5)
        lo, hi = err(row["v"], row["lo"], row["hi"])
        ax.errorbar(i, row["v"], yerr=[[lo], [hi]], fmt="none", ecolor="black", elinewidth=1.6, capsize=5)
        # Recall, range, edits found and the false-region count, printed together above the bar.
        ax.annotate(f"recall {fmt(row)}\n{row['n_f']:,} of {row['n_e']:,} edits found\n"
                    f"FALSE REGIONS: {row['n_fr']:,}", (i, row["hi"]), ha="center", va="bottom", fontsize=8.5,
                    xytext=(0, 6), textcoords="offset points")
    n = s[("full", "all", "all", "loc_recall")]
    ax.set_xticks(range(2))
    ax.set_xticklabels(SIZES)
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Location recall: edits found / all edits (0 to 1)")
    ax.set_xlabel("Input size")
    ax.set_title("ADCD-Net location recall with its false-region count (95% range)", fontsize=10.5)
    ax.text(0.5, -0.17, f"n = {n['n_t']:,} edited pages ({n['n_e']:,} edits) and {n['n_u']:,} untouched pages; "
            "false regions counted on all pages", transform=ax.transAxes, ha="center", fontsize=7.5)
    style_axes(ax)
    return save_png(fig, out, "fig3_location")


# --------------------------------------------------------------------------- figure 4
def fig4(s, d, out):
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 5.4), gridspec_kw={"width_ratios": [1, 1, 1.05]})
    offs = {"ai": -0.1, "handcrafted": 0.1}
    side = {"ai": ("right", -9), "handcrafted": ("left", 9)}
    # Panels 1 and 2: AUC and location recall by group, both sizes.
    for ax, metric, ylab, title in ((axes[0], "auc", "AUC (0 to 1)", "AUC"),
                                    (axes[1], "loc_recall", "Location recall (0 to 1)", "Location recall")):
        for g in ("ai", "handcrafted"):
            for i, size in enumerate(SIZES):
                row = s[(size, "group", g, metric)]
                errbars(ax, i + offs[g], row, GROUP_MARK[g], GROUP_COLOUR[g], size,
                        label=f"{GROUP_NAME[g]} ({row['n_t']} edited pages, {row['n_e']} edits), {size}"
                        if metric == "auc" else None)
                text = f"{row['v']:.3f}\n({row['lo']:.3f} to {row['hi']:.3f})"
                if metric == "loc_recall":
                    text += f"\n{row['n_f']} of {row['n_e']} found\nfalse regions: {row['n_fr']:,}"
                # AI label goes left of its point and handcrafted right, so labels never overlap.
                ax.annotate(text, (i + offs[g], row["v"]), ha=side[g][0], va="center", fontsize=6.8,
                            xytext=(side[g][1], 0), textcoords="offset points")
        ax.set_xticks(range(2))
        ax.set_xticklabels(SIZES)
        ax.set_xlim(-0.85, 1.85)
        ax.set_ylabel(ylab)
        ax.set_title(f"{title} by group", fontsize=10.5)
        style_axes(ax)
    axes[0].axhline(0.5, color="black", linestyle=":", linewidth=1.2, label="chance (AUC = 0.5)")
    axes[0].set_ylim(0.4, 0.7)
    axes[0].set_ylabel("AUC (0 to 1; axis shows 0.4 to 0.7 only)")
    axes[0].legend(fontsize=6.3, loc="lower right", framealpha=0.95)
    axes[1].set_ylim(0, 0.95)
    # Panel 3: difference (AI minus handcrafted) with its 95% range, zero line.
    ax = axes[2]
    for j, (metric, name) in enumerate((("auc", "AUC difference"), ("loc_recall", "Location recall difference"))):
        for i, size in enumerate(SIZES):
            diff, lo, hi = d[("ai minus handcrafted", size, "group", metric)]
            x = j * 2.0 + i * 0.7
            style = SIZE_STYLE[size]
            ax.errorbar(x, diff, yerr=[[diff - lo], [hi - diff]], fmt="D", color="black",
                        markerfacecolor="black" if style["filled"] else "white", markersize=8, markeredgewidth=1.6,
                        elinewidth=1.6, capsize=4, linestyle="none", label=size if j == 0 else None)
            # Label above the range for positive differences, below it for negative ones (keeps the zero line clear).
            below = diff < 0
            ax.annotate(f"{diff:+.3f}\n({lo:+.3f} to {hi:+.3f})", (x, lo if below else hi), ha="center",
                        va="top" if below else "bottom", fontsize=7.5, xytext=(0, -5 if below else 5),
                        textcoords="offset points")
    ax.axhline(0, color="black", linestyle="-", linewidth=1.0)
    ax.set_xticks([0.35, 2.35])
    ax.set_xticklabels(["AUC", "Location recall"])
    ax.set_xlim(-0.6, 3.0)
    ax.set_ylim(-0.22, 0.22)
    ax.set_ylabel("AI minus handcrafted (difference in score)")
    ax.set_title("Difference, AI minus handcrafted (95% range)", fontsize=10.5)
    style_axes(ax)
    ax.legend(fontsize=7.5, loc="lower right", framealpha=0.95)
    fig.suptitle("ADCD-Net, AI-edited vs handcrafted pages, both input sizes (hollow = shrink512, filled = full)",
                 fontsize=11.5, y=1.0)
    fig.text(0.5, -0.02, "Untouched pages for AUC: 644 at both sizes. Location recall always printed with the "
             "false-region count of that group.", ha="center", fontsize=8)
    fig.tight_layout()
    return save_png(fig, out, "fig4_ai_vs_hand")


# --------------------------------------------------------------------------- figure 5
def fig5(s, out):
    panels = [
        ("editor_or_method", ["AnyText", "TextCtrl", "UDiffText", "retypeset", "copymove", "pdf_edit"],
         "By editor / method"),
        ("capture", ["native", "jpeg", "resize_jpeg", "scan_sim"], "By capture condition"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.8), gridspec_kw={"width_ratios": [1.15, 1]})
    for ax, (breakdown, groups, title) in zip(axes, panels):
        for k, g in enumerate(groups):
            for size, off in zip(SIZES, (-0.15, 0.15)):
                row = s[(size, breakdown, g, "auc")]
                y = k + off
                errbars(ax, y, row, "o" if size == "full" else "s", "black", size, horizontal=True, ms=7,
                        label=size if k == 0 else None)
                ax.annotate(f"{row['v']:.3f} ({row['lo']:.3f} to {row['hi']:.3f})", (row["hi"], y), va="center",
                            fontsize=6.8, xytext=(5, 0), textcoords="offset points")
        ax.set_yticks(range(len(groups)))
        # n edited pages in each group, on the label; untouched pages are the same pool for every group.
        ax.set_yticklabels([f"{g}\nn = {s[('full', breakdown, g, 'auc')]['n_t']} edited pages" for g in groups],
                           fontsize=8)
        ax.invert_yaxis()
        ax.axvline(0.5, color="black", linestyle=":", linewidth=1.2, label="chance (AUC = 0.5)")
        ax.set_xlim(0.38, 0.86)
        ax.set_xlabel("AUC (0 to 1; axis shows 0.38 to 0.86 only)")
        ax.set_title(title, fontsize=10.5)
        ax.grid(axis="x", color="#cccccc", linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.legend(fontsize=7.5, loc="upper right", framealpha=0.95)
    # The untouched-page count differs by capture condition, so print it for that panel.
    caps = ["native", "jpeg", "resize_jpeg", "scan_sim"]
    cap_txt = "; ".join(f"{c} {s[('full', 'capture', c, 'auc')]['n_u']}" for c in caps)
    fig.suptitle("ADCD-Net AUC by editor/method and by capture condition, both input sizes (95% ranges)",
                 fontsize=11.5, y=1.0)
    fig.text(0.5, -0.03, "Untouched pages: 644 in every editor/method group. Untouched pages by capture condition: "
             + cap_txt + ". Groups are shown side by side, not tested against each other.", ha="center",
             fontsize=8)
    fig.tight_layout()
    return save_png(fig, out, "fig5_breakdowns")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", type=Path, default=REPO / "results" / "adcd")
    parser.add_argument("--out-dir", type=Path, default=REPO / "results" / "adcd" / "figures")
    args = parser.parse_args()
    s = load_scores(args.results_dir / "scores.csv")
    d = load_diffs(args.results_dir / "differences.csv")
    for path in (fig1(s, args.out_dir), fig2(s, args.out_dir), fig3(s, args.out_dir), fig4(s, d, args.out_dir),
                 fig5(s, args.out_dir)):
        print("wrote", path)


if __name__ == "__main__":
    main()
