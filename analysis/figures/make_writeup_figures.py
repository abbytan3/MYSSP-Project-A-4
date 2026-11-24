"""Figures 2 to 5 of the write-up, drawn with matplotlib from the committed score tables.

This is the script that drew the write-up's Figures 2-5 (write-up chat, 23 Nov 2026, version 7.1), committed so the
figures can be redrawn from this repo. Only three things differ from the run that made the submitted figures, and
none changes a plotted value: the input folder (`results/` of this repo instead of a scratch copy of it), the output
folder (an argument), and a fallback font when TeX Gyre Termes is not installed. Figure 1 (the pipeline) is drawn
with TikZ inside the LaTeX source, not here.

Reads:  results/<detector>/scores.csv for adcd, dtd, daf_anytext, daf_diffste (rows with breakdown = all)
        results/human/auc.csv
Writes: <out-dir>/fig2_auc.png, fig3_ruleB.png, fig4_location.png, fig5_people.png (300 dpi)
        <out-dir>/figure_data.csv  every plotted value with its 95% CI (67 rows), for checking against the write-up
        PDFs of the four figures too with --pdf (the write-up used the PDFs; PDFs never go in this repo)

Usage, from the repo root:
  python3 analysis/figures/make_writeup_figures.py                                   # PNGs + figure_data.csv into results/writeup/figures/
  python3 analysis/figures/make_writeup_figures.py --out-dir <folder outside the repo> --pdf
"""
# read the command-line options
import argparse
# read and write the CSV tables
import csv
# an in-memory file for the PDF draw when no PDF is kept
import io
# build paths that work on Mac and Windows
from pathlib import Path
# matplotlib draws every figure; Agg draws to files without opening a window
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# the repo root is two folders up from this file
REPO = Path(__file__).resolve().parents[2]
# command-line options: where the figures go, and whether to also write PDFs
parser = argparse.ArgumentParser(description="Draw the write-up's Figures 2-5 from results/.")
parser.add_argument("--out-dir", type=Path, default=REPO / "results" / "writeup" / "figures")
parser.add_argument("--pdf", action="store_true", help="also write PDFs (keep them outside the repo)")
args = parser.parse_args()
# every input table is read from this repo's results folder
R = REPO / "results"
OUT = args.out_dir
OUT.mkdir(parents=True, exist_ok=True)

# the write-up's font (TeX Gyre Termes, a Times-like font that ships with TeX Live); register it if it is installed
for f in fm.findSystemFonts(["/usr/share/texmf/fonts/opentype/public/tex-gyre", "/usr/share/texlive/texmf-dist/fonts/opentype/public/tex-gyre", "/Library/TeX/Root/texmf-dist/fonts/opentype/public/tex-gyre"]):
    fm.fontManager.addfont(f)
# use it if it was found, otherwise matplotlib's default serif font (only the lettering changes)
FONT = "TeX Gyre Termes" if any(f.name == "TeX Gyre Termes" for f in fm.fontManager.ttflist) else "serif"
if FONT == "serif":
    print("note: TeX Gyre Termes not found, using matplotlib's default serif font; plotted values are unchanged")
# the same style settings as the write-up figures
plt.rcParams.update({"font.family": FONT, "font.size": 8, "axes.titlesize": 8,
    "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "axes.edgecolor": "#555555",
    "xtick.color": "#333333", "ytick.color": "#333333", "axes.labelcolor": "#222222",
    "pdf.fonttype": 42})
W = 5.5                       # text width of the paper in inches; every figure uses it
DET = [("adcd", "ADCD-Net"), ("dtd", "DTD"), ("daf_anytext", "DAF (AnyText weight)"), ("daf_diffste", "DAF (DiffSTE weight)")]
COL = {"adcd": "#2a78d6", "dtd": "#e87ba4", "daf_anytext": "#eb6834", "daf_diffste": "#1baf7a"}
MRK = {"adcd": "o", "dtd": "D", "daf_anytext": "s", "daf_diffste": "^"}
SIZES = [("shrink512", "Downscaled"), ("full", "Full")]
GRID = "#e6e6e6"
LOG = []  # every plotted value, written to figure_data.csv for checking

def rows(det):
    return list(csv.DictReader(open(R / det / "scores.csv", encoding="utf-8")))
def get(det, size, metric):
    for r in rows(det):
        if r["size"] == size and r["breakdown"] == "all" and r["metric"] == metric:
            return r
    raise KeyError((det, size, metric))
def save(fig, name):
    # The PDF is always drawn first, as in the run that made the paper's figures: with constrained layout the first
    # draw settles the layout, so skipping it would shift Figure 3 slightly. It is kept only with --pdf (the NDA hook
    # keeps PDFs out of the repo); otherwise it is drawn to memory and dropped. The PNG (300 dpi) is always kept.
    fig.savefig(OUT / f"{name}.pdf" if args.pdf else io.BytesIO(), format="pdf")
    fig.savefig(OUT / f"{name}.png", dpi=300); plt.close(fig)
def size_legend(ax, **kw):
    h = [Line2D([], [], marker="o", ls="none", mfc="white", mec="#333333", label="Downscaled"),
         Line2D([], [], marker="o", ls="none", mfc="#333333", mec="#333333", label="Full")]
    return ax.legend(handles=h, frameon=False, **kw)

# Figure 2: AUC per detector and input size
fig, ax = plt.subplots(figsize=(W, 2.4), layout="constrained")
for i, (d, name) in enumerate(DET):
    for j, (s, sname) in enumerate(SIZES):
        r = get(d, s, "auc"); v, lo, hi = float(r["value"]), float(r["ci_low"]), float(r["ci_high"])
        x = i + (j - 0.5) * 0.28
        LOG.append(("fig2", d, s, "auc", v, lo, hi))
        ax.errorbar(x, v, yerr=[[v - lo], [hi - v]], fmt=MRK[d], color=COL[d], mfc="white" if j == 0 else COL[d],
                    mec=COL[d], ms=6, capsize=3, elinewidth=1, mew=1.2)
        ax.annotate(f"{v:.3f}", (x, hi), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=6.5, color="#333333")
ax.axhline(0.5, ls="--", lw=0.8, color="#777777")
ax.text(3.62, 0.5, "AUC = 0.5", va="bottom", ha="right", fontsize=7, color="#555555")
ax.set_xticks(range(4)); ax.set_xticklabels([n for _, n in DET]); ax.set_xlim(-0.5, 3.65)
ax.set_ylim(0.45, 0.65); ax.set_ylabel("AUC (95% CI)"); ax.yaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
size_legend(ax, loc="upper left", ncol=2)
save(fig, "fig2_auc")

# Figure 3: rule B recall against false-alarm rate (full range and zoomed)
fig, axs = plt.subplots(1, 2, figsize=(W, 3.0), gridspec_kw={"width_ratios": [1, 1.25]}, layout="constrained")
for k, ax in enumerate(axs):
    for d, name in DET:
        for j, (s, _) in enumerate(SIZES):
            rc, fa = get(d, s, "recall_B"), get(d, s, "false_alarm_B")
            x, y = float(fa["value"]), float(rc["value"])
            if k == 0: LOG += [("fig3", d, s, "recall_B", y, float(rc["ci_low"]), float(rc["ci_high"])), ("fig3", d, s, "false_alarm_B", x, float(fa["ci_low"]), float(fa["ci_high"]))]
            ax.errorbar(x, y, xerr=[[x - float(fa["ci_low"])], [float(fa["ci_high"]) - x]],
                        yerr=[[y - float(rc["ci_low"])], [float(rc["ci_high"]) - y]], fmt=MRK[d], color=COL[d],
                        mfc="white" if j == 0 else COL[d], mec=COL[d], ms=5 if k else 4, capsize=2 if k else 0, elinewidth=0.8, mew=1)
    lim = (0, 1.0) if k == 0 else (0.97, 1.003)
    ax.plot(lim, lim, ls="--", lw=0.8, color="#777777")
    ax.set_xlim(lim[0] - (0.02 if k == 0 else 0), lim[1] + (0.02 if k == 0 else 0)); ax.set_ylim(*ax.get_xlim())
    ax.set_xlabel("False-alarm rate on untouched pages"); ax.set_ylabel("Recall on edited pages")
    ax.set_title("Full range" if k == 0 else "Zoomed to the points"); ax.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.set_aspect("equal")
axs[0].text(0.55, 0.42, "recall = false-alarm rate", rotation=45, fontsize=7, color="#555555", ha="center")
h = [Line2D([], [], marker=MRK[d], ls="none", color=COL[d], label=n) for d, n in DET]
h += [Line2D([], [], marker="o", ls="none", mfc="white", mec="#333333", label="Downscaled"),
      Line2D([], [], marker="o", ls="none", mfc="#333333", mec="#333333", label="Full")]
fig.legend(handles=h, loc="outside lower center", ncol=3, frameon=False)
save(fig, "fig3_ruleB")

# Figure 4: location recall and false regions
fig, axs = plt.subplots(1, 2, figsize=(W, 2.7), layout="constrained")
bw = 0.36
for i, (d, name) in enumerate(DET):
    for j, (s, _) in enumerate(SIZES):
        r = get(d, s, "loc_recall"); v = float(r["value"]); x = i + (j - 0.5) * bw
        kw = dict(color="white" if j == 0 else COL[d], edgecolor=COL[d], hatch="////" if j == 0 else None, linewidth=1)
        axs[0].bar(x, v, bw * 0.9, **kw)
        axs[0].errorbar(x, v, yerr=[[v - float(r["ci_low"])], [float(r["ci_high"]) - v]], fmt="none", ecolor="#333333", capsize=2, elinewidth=0.8)
        fr = int(r["n_false_regions"])
        LOG += [("fig4", d, s, "loc_recall", v, float(r["ci_low"]), float(r["ci_high"])), ("fig4", d, s, "false_regions", fr, "", "")]
        axs[1].bar(x, fr, bw * 0.9, **kw)
        axs[1].annotate(f"{fr:,}", (x, fr), textcoords="offset points", xytext=(0, 2), ha="center", va="bottom", rotation=90, fontsize=6.5, color="#333333")
for ax in axs:
    ax.set_xticks(range(4)); ax.set_xticklabels(["ADCD-Net", "DTD", "DAF\n(AnyText)", "DAF\n(DiffSTE)"])
    ax.yaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
axs[0].set_ylim(0, 1.05); axs[0].set_ylabel("Location recall (95% CI)"); axs[0].set_title("Edits found (of 1,093)")
axs[1].set_yscale("log"); axs[1].set_ylim(1e3, 3e6); axs[1].set_ylabel("False regions (log scale)"); axs[1].set_title("False regions on all 1,650 pages")
h = [Patch(facecolor="white", edgecolor="#333333", hatch="////", label="Downscaled"), Patch(facecolor="#333333", edgecolor="#333333", label="Full")]
fig.legend(handles=h, loc="outside lower center", ncol=2, frameon=False)
save(fig, "fig4_location")

# Figure 5: people and detectors on the 500 rated pages
H = list(csv.DictReader(open(R / "human" / "auc.csv", encoding="utf-8")))
def h(comp, scorer):
    for r in H:
        if r["comparison"] == comp and r["scorer"] == scorer:
            return float(r["auc"]), float(r["ci_low"]), float(r["ci_high"])
    raise KeyError((comp, scorer))
SC = [("people_mean", "People (mean of 2 raters)", "#222222", "o", True)]
for d, n in DET:
    for s, sn in SIZES:
        SC.append((f"{d}_{s}", f"{n}, {sn}", COL[d], MRK[d], s == "full"))
COMP = [("edited_vs_authentic", "All edited (320 vs 180)"), ("ai_vs_authentic", "AI-edited (160 vs 180)"), ("handcrafted_vs_authentic", "Handcrafted (160 vs 180)")]
fig, axs = plt.subplots(1, 3, figsize=(W, 3.0), sharey=True, layout="constrained")
for k, (comp, title) in enumerate(COMP):
    ax = axs[k]
    for y, (sc, lab, c, m, filled) in enumerate(SC):
        v, lo, hi = h(comp, sc)
        LOG.append(("fig5", sc, comp, "auc", v, lo, hi))
        ax.errorbar(v, -y, xerr=[[v - lo], [hi - v]], fmt=m, color=c, mfc=c if filled else "white", mec=c, ms=4.5, capsize=1.5, elinewidth=0.8, mew=1)
    ax.axvline(0.5, ls="--", lw=0.8, color="#777777")
    ax.set_xlim(0.35, 1.0); ax.set_title(title); ax.set_xlabel("AUC (95% CI)")
    ax.xaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
axs[0].set_yticks([-y for y in range(len(SC))]); axs[0].set_yticklabels([s[1] for s in SC])
axs[0].tick_params(axis="y", length=0)
save(fig, "fig5_people")
# write every plotted value next to the figures, so each can be checked against the write-up
with open(OUT / "figure_data.csv", "w", encoding="utf-8", newline="") as fh:
    csv.writer(fh).writerows([("figure", "series", "size_or_comparison", "metric", "value", "ci_low", "ci_high")] + LOG)
print("ok", len(LOG))
