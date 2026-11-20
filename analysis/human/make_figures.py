"""T49: two charts from the tables in results/human/ (numbers only; nothing else is read).

    python3 analysis/human/make_figures.py

Writes PNG (300 dpi) into results/human/figures/. Greyscale-readable: scorers differ by
marker shape and filled/hollow, not only by colour. Each chart is drawn only from the CSV
tables, so anyone can rerun this and get the same picture.
"""

import csv
from pathlib import Path

import matplotlib

# Draw to files, no window.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
TABLES = REPO / "results" / "human"
FIGURES = TABLES / "figures"

# Recessive ink colours for text, axes and grid.
INK, MUTED, GRID = "#222222", "#666666", "#dddddd"
# One style per scorer: marker shape, filled or hollow, colour. People are filled, detectors hollow.
PEOPLE_STYLE = {"people_mean": ("o", True, "#1b1b1b"), "people_abby": ("s", True, "#6b6b6b"),
                "people_nazhan": ("D", True, "#9a9a9a")}
DETECTOR_MARKERS = ["^", "v", "P", "X", "*", "h", "<", ">"]
DETECTOR_COLOURS = ["#0b5cad", "#c05a00", "#2a7f3f", "#8a2c8a", "#a08000", "#007a7a", "#b3203c", "#5a5a00"]
# Plain-words labels for the legend and rows.
COMPARISON_LABEL = {"edited_vs_authentic": "All edited vs authentic", "ai_vs_authentic": "AI-edited vs authentic",
                    "handcrafted_vs_authentic": "Handcrafted vs authentic"}
PEOPLE_LABEL = {"people_mean": "People (mean of 2)", "people_abby": "Abby alone", "people_nazhan": "Nazhan alone"}
VERDICT_LABEL = {"people_flagged": "people flagged", "in_between": "in between", "people_missed": "people missed (both gave 5)"}


def read_table(name):
    """Read one results table as a list of dicts."""
    with open(TABLES / name, encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def detector_styles(detector_sizes):
    """One marker and colour per detector/size, in a fixed order so a figure never repaints."""
    styles = {}
    for i, key in enumerate(sorted(detector_sizes)):
        styles[key] = (DETECTOR_MARKERS[i % len(DETECTOR_MARKERS)], False, DETECTOR_COLOURS[i % len(DETECTOR_COLOURS)])
    return styles


def tidy(ax):
    """Quiet axes: light horizontal grid only, no top/right frame."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=INK, labelsize=9)
    ax.xaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def draw_point(ax, x, y, lo, hi, style, label=None, small=False):
    """One estimate with its 95% range. Hollow marker for detectors; dotted range if the n is under 30."""
    marker, filled, colour = style
    ax.plot([lo, hi], [y, y], color=colour, linewidth=1.4, linestyle=":" if small else "-", solid_capstyle="butt", zorder=2)
    ax.plot([x], [y], marker=marker, markersize=7, markerfacecolor=colour if filled else "white",
            markeredgecolor=colour, markeredgewidth=1.4, linestyle="none", label=label, zorder=3)


def figure_auc(path):
    """Chart 1: AUC for people and detectors, three comparisons, each with its 95% range."""
    rows = read_table("auc.csv")
    scorers = [r["scorer"] for r in rows if r["comparison"] == "edited_vs_authentic"]
    det_styles = detector_styles({r["scorer"] for r in rows if r["source"] == "detector"})
    styles = {**PEOPLE_STYLE, **det_styles}
    comps = list(COMPARISON_LABEL)
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    step, gap = 1.0, 0.8
    y, ticks, labels, seen = 0.0, [], [], set()
    for comp in comps:
        first_y = y
        for name in scorers:
            r = next(x for x in rows if x["comparison"] == comp and x["scorer"] == name)
            label = None if name in seen else (PEOPLE_LABEL.get(name) or name.replace("_", " "))
            seen.add(name)
            draw_point(ax, float(r["auc"]), y, float(r["ci_low"]), float(r["ci_high"]), styles[name], label,
                       small=r["small_n"] == "True")
            y += step
        n_e, n_a = r["n_edited"], r["n_authentic"]
        ticks.append((first_y + y - step) / 2)
        labels.append(f"{COMPARISON_LABEL[comp]}\n(n = {n_e} vs {n_a})")
        y += gap
    ax.axvline(0.5, color=MUTED, linewidth=1, linestyle="--")
    ax.annotate(" chance", xy=(0.5, 0.99), xycoords=("data", "axes fraction"), color=MUTED, fontsize=8, ha="left", va="top")
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(0.3, 1.0)
    ax.set_xlabel("AUC (0.5 = chance, 1 = every edited page ranked above every authentic page)", color=INK, fontsize=9)
    ax.set_title("Ranking edited above authentic pages: people vs detectors (95% ranges)", color=INK, fontsize=11, loc="left")
    tidy(ax)
    ax.legend(loc="lower right", fontsize=8, frameon=False, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def figure_by_people(path):
    """Chart 2: each detector's AUC on edits people flagged, missed or in between (against all authentic pages)."""
    rows = read_table("detector_auc_by_people.csv")
    keys = sorted({(r["detector"], r["size"]) for r in rows})
    det_styles = detector_styles({f"{d}_{s}" for d, s in keys})
    scopes = [("edited", "All edited"), ("ai", "AI-edited"), ("handcrafted", "Handcrafted")]
    verdicts = ["people_flagged", "in_between", "people_missed"]
    fig, ax = plt.subplots(figsize=(8.2, 7.6))
    y, ticks, labels = 0.0, [], []
    for scope, scope_label in scopes:
        for verdict in verdicts:
            for k, (det, size) in enumerate(keys):
                r = next(x for x in rows if x["detector"] == det and x["size"] == size
                         and x["edited_pages"] == scope and x["people"] == verdict)
                small = r["small_n"] == "True"
                draw_point(ax, float(r["auc"]), y + 0.18 * (k - (len(keys) - 1) / 2), float(r["ci_low"]), float(r["ci_high"]),
                           det_styles[f"{det}_{size}"], f"{det} {size}" if (scope, verdict) == ("edited", verdicts[0]) else None,
                           small=small)
            ticks.append(y)
            labels.append(f"{scope_label}, {VERDICT_LABEL[verdict]}\n(n = {r['n_edited']}{', shaky' if small else ''})")
            y += 1.0
        y += 0.5
    ax.axvline(0.5, color=MUTED, linewidth=1, linestyle="--")
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Detector AUC against the 180 authentic pages (0.5 = chance)", color=INK, fontsize=9)
    ax.set_title("Detector ranking, by what people made of the edit (95% ranges)", color=INK, fontsize=11, loc="left")
    tidy(ax)
    # Legend: the detector sizes plus a proxy line explaining the dotted ranges.
    handles, names = ax.get_legend_handles_labels()
    handles.append(Line2D([0], [0], color=MUTED, linestyle=":", linewidth=1.4))
    names.append("dotted = fewer than 30 edited pages (shaky, do not interpret)")
    # Legend sits below the plot so it never covers a point.
    ax.legend(handles, names, loc="upper center", bbox_to_anchor=(0.4, -0.1), ncol=3, fontsize=8, frameon=False, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    figure_auc(FIGURES / "auc_people_vs_detectors.png")
    figure_by_people(FIGURES / "detector_auc_by_people_verdict.png")
    print(f"wrote 2 PNG charts to {FIGURES.relative_to(REPO)}")


if __name__ == "__main__":
    main()
