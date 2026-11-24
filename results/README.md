# results/: what each file holds and how to read it

Every number the project reports comes from a file in this folder or in `analysis/ablation/results/`. The files hold numbers and anonymous IDs only: no pages, file names or statement text. This guide was written on 24 Nov 2026; every value quoted below was read from the CSV files by script and rounded to 3 decimals.

Brackets hold 95% confidence intervals (CIs): the 2.5th and 97.5th percentiles of 1,000 bootstrap resamples of documents (pages of one document stay together), seed 42. "Downscaled" is `shrink512` in the files (longest side 512 px); "Full" is `full` (full size in tiles). Other terms are defined in the main README, "Terms used in this repo".

## Where things are

| Folder | Files | Made by |
|---|---|---|
| `adcd/`, `dtd/`, `daf_anytext/`, `daf_diffste/` | `scores.csv` (every metric with its CI and counts, overall and by group, editor or method, and capture condition), `differences.csv` (AI minus handcrafted; downscaled minus full; paired CIs), `score_info.json` (seed, rule B cut-offs per fold, missing pages, a copy of the run's `run_info.json`) | `shared/score.py` |
| `adcd/figures/` | 5 charts and `SUMMARY.md` (what each chart shows) | `analysis/figures/make_figures_t16.py` |
| `daf_anytext/figures/`, `daf_diffste/figures/` | 5 charts each (same layout as ADCD-Net's) | `analysis/figures/make_figures_t18.py` |
| `dtd/` | `SUMMARY.md` (tables only, no charts) and `top_score_spread.csv` | `analysis/dtd/top_score_spread.py` |
| `combined/tables/` | `detector_table.csv`, `ai_minus_handcrafted.csv`, `size_difference.csv`, `people_vs_detectors.csv`, `human_grade_breakdown.csv` (values copied as text, no rounding) | `analysis/combined/make_tables.py` |
| `combined/figures/` | `fig_auc.png`, `fig_ruleB_recall_vs_fa.png`, `fig_location.png` | `analysis/figures/make_figures.py` |
| `human/` | People vs detectors on the 500 real stage 2 pages (see `analysis/human/README.md` for each file) | `analysis/human/people_vs_detectors.py` |
| `human/figures/` | `auc_people_vs_detectors.png`, `detector_auc_by_people_verdict.png` | `analysis/human/make_figures.py` |
| `writeup/figures/` | The write-up's Figures 2 to 5 (PNG, 300 dpi) and `figure_data.csv` (all 67 plotted values with 95% CIs) | `analysis/figures/make_writeup_figures.py` |
| `adcd/tile_check/` | Printed output of the page-size checks: `tile_count_vs_label.txt` (AUC of tile count and page area alone, and ADCD-Net's AUC inside tile-count groups) and `tile_count_vs_top.txt` (top score against tile count) | `analysis/tile_check/` scripts, run on the private ADCD-Net outputs |
| `../analysis/ablation/results/` | `ablation_scores.csv` (240 rows) and `contrast_summary.csv` (148 paired contrasts) | `analysis/ablation/` scripts |

In `scores.csv`, a row is one detector, size, `breakdown` (`all`, `group`, `editor_or_method`, `capture`), `subset` and `metric`. Metrics: `auc`; rule A `precision_A`, `recall_A`, `f1_A`, `false_alarm_A`; rule B `precision_B`, `recall_B`, `f1_B`, `false_alarm_B`; location `loc_recall`, `loc_precision`, `loc_f1`. Each row also holds `n_tampered`, `n_untouched`, `n_edits`, `n_found` and `n_false_regions`. All four `score_info.json` files report 0 missing pages at both sizes.

## RQ1: detection (1,006 edited pages vs 644 untouched pages)

| Detector | Input | AUC, all edited | AUC, AI-edited | AUC, handcrafted | Rule A false-alarm rate | Rule B false-alarm rate |
|---|---|---|---|---|---|---|
| ADCD-Net | Downscaled | 0.503 [0.486, 0.522] | 0.520 [0.497, 0.543] | 0.487 [0.465, 0.511] | 0.693 [0.655, 0.727] | 0.995 [0.991, 1.000] |
| ADCD-Net | Full | 0.533 [0.510, 0.557] | 0.559 [0.531, 0.585] | 0.506 [0.477, 0.535] | 1.000 [1.000, 1.000] | 0.986 [0.975, 0.994] |
| DTD | Downscaled | 0.507 [0.482, 0.532] | 0.516 [0.486, 0.544] | 0.498 [0.470, 0.525] | 1.000 [1.000, 1.000] | 0.995 [0.989, 1.000] |
| DTD | Full | 0.505 [0.478, 0.532] | 0.535 [0.505, 0.564] | 0.475 [0.444, 0.506] | 0.829 [0.800, 0.857] | 0.997 [0.992, 1.000] |
| DAF (AnyText weight) | Downscaled | 0.505 [0.487, 0.522] | 0.509 [0.486, 0.533] | 0.501 [0.481, 0.521] | 0.998 [0.995, 1.000] | 0.995 [0.989, 1.000] |
| DAF (AnyText weight) | Full | 0.598 [0.577, 0.622] | 0.710 [0.682, 0.738] | 0.484 [0.459, 0.508] | 0.913 [0.890, 0.933] | 0.991 [0.983, 0.997] |
| DAF (DiffSTE weight) | Downscaled | 0.512 [0.492, 0.530] | 0.515 [0.490, 0.540] | 0.509 [0.484, 0.530] | 0.989 [0.980, 0.995] | 1.000 [1.000, 1.000] |
| DAF (DiffSTE weight) | Full | 0.595 [0.574, 0.618] | 0.698 [0.673, 0.727] | 0.490 [0.464, 0.518] | 0.994 [0.988, 0.998] | 1.000 [1.000, 1.000] |

- AI-edited: 506 pages; handcrafted: 500 pages; each compared with the same 644 untouched pages.
- Rule B's cut-off maximises F1, and with weak scores that cut-off falls low enough to call almost every page edited. Its F1 (about 0.75) is roughly what flagging every page gives when 1,006 of 1,650 pages are edited, so it says nothing on its own. Read rule B only with its false-alarm rate, and lead with AUC.
- AI minus handcrafted AUC and downscaled minus full AUC, with paired CIs, are in `combined/tables/ai_minus_handcrafted.csv` and `combined/tables/size_difference.csv` (the size file keeps the stored direction, downscaled minus full; do not flip it by hand).

## RQ2: localisation (1,093 edits on 1,006 edited pages; false regions counted on all 1,650 pages)

| Detector | Input | Location recall | Edits found | Location precision | False regions |
|---|---|---|---|---|---|
| ADCD-Net | Downscaled | 0.101 [0.083, 0.120] | 110 of 1,093 | 0.013 [0.011, 0.015] | 8,419 |
| ADCD-Net | Full | 0.612 [0.581, 0.645] | 669 of 1,093 | 0.003 [0.002, 0.003] | 262,769 |
| DTD | Downscaled | 0.980 [0.967, 0.990] | 1,071 of 1,093 | 0.105 [0.100, 0.110] | 9,164 |
| DTD | Full | 0.366 [0.332, 0.403] | 400 of 1,093 | 0.011 [0.010, 0.013] | 34,754 |
| DAF (AnyText weight) | Downscaled | 0.156 [0.135, 0.179] | 171 of 1,093 | 0.010 [0.008, 0.011] | 17,517 |
| DAF (AnyText weight) | Full | 0.338 [0.310, 0.366] | 369 of 1,093 | 0.043 [0.039, 0.047] | 8,286 |
| DAF (DiffSTE weight) | Downscaled | 0.284 [0.256, 0.314] | 310 of 1,093 | 0.007 [0.007, 0.008] | 41,223 |
| DAF (DiffSTE weight) | Full | 0.404 [0.376, 0.433] | 442 of 1,093 | 0.018 [0.017, 0.020] | 23,714 |

- Never quote location recall without the false-region count. High recall here comes from flagging large parts of each page (DTD downscaled flags every page), not from finding edits.
- The AI and handcrafted rows in `scores.csv` each include the false regions on all 644 untouched pages, so the two groups' false-region counts do not add up to the overall count.

## RQ3: conditions

**Editor, method and capture condition** (`breakdown` = `editor_or_method` or `capture` in `scores.csv`). Each editor or method row compares its edited pages with all 644 untouched pages; each capture row uses only the edited and untouched pages of that condition. Group sizes (edited / untouched): AnyText 177 / 644, UDiffText 204 / 644, TextCtrl 125 / 644, retypeset 333 / 644, copy-move 95 / 644, PDF edit 72 / 644; native 181 / 115, jpeg 409 / 249, resize_jpeg 205 / 150, scan_sim 211 / 130.

Pairwise differences between groups were **not** computed (`differences.csv` has none). A script compared every pair of 95% CIs within capture conditions, within AI editors and within handcrafted methods, for each detector and size; these 11 pairs are the only ones whose CIs do not overlap. Describe them as "the intervals do not overlap, which suggests …", never as a significant difference.

| Detector | Input | Group A | Group B |
|---|---|---|---|
| DAF (AnyText weight) | Full | native 0.702 [0.647, 0.752] | resize_jpeg 0.511 [0.450, 0.572] |
| DAF (AnyText weight) | Full | native 0.702 [0.647, 0.752] | scan_sim 0.527 [0.463, 0.595] |
| DAF (AnyText weight) | Full | jpeg 0.636 [0.596, 0.673] | resize_jpeg 0.511 [0.450, 0.572] |
| DAF (AnyText weight) | Full | jpeg 0.636 [0.596, 0.673] | scan_sim 0.527 [0.463, 0.595] |
| DAF (DiffSTE weight) | Full | native 0.679 [0.623, 0.733] | scan_sim 0.463 [0.407, 0.530] |
| DAF (DiffSTE weight) | Full | jpeg 0.628 [0.592, 0.668] | scan_sim 0.463 [0.407, 0.530] |
| DAF (AnyText weight) | Full | AnyText 0.644 [0.602, 0.688] | UDiffText 0.732 [0.692, 0.773] |
| DAF (AnyText weight) | Full | AnyText 0.644 [0.602, 0.688] | TextCtrl 0.770 [0.727, 0.810] |
| DAF (DiffSTE weight) | Full | AnyText 0.644 [0.603, 0.692] | TextCtrl 0.754 [0.707, 0.801] |
| ADCD-Net | Downscaled | AnyText 0.470 [0.432, 0.508] | UDiffText 0.561 [0.522, 0.596] |
| DAF (AnyText weight) | Downscaled | AnyText 0.477 [0.428, 0.519] | TextCtrl 0.568 [0.520, 0.613] |

- Capture groups mix AI-edited and handcrafted pages, so a different share of AI pages in each condition could produce part of the gap.
- The two downscaled pairs sit close to 0.5 on both sides; treat them as weak.
- ADCD-Net and DTD: every capture-condition pair overlaps at both sizes.

**People vs detectors** (500 real stage 2 pages: 160 AI-edited, 160 handcrafted, 180 authentic; source `human/auc.csv`):

| Scorer | Edited vs authentic (320 / 180) | AI-edited vs authentic (160 / 180) | Handcrafted vs authentic (160 / 180) |
|---|---|---|---|
| People (mean of two raters) | 0.748 [0.708, 0.787] | 0.930 [0.900, 0.958] | 0.567 [0.511, 0.622] |
| ADCD-Net, Downscaled | 0.470 [0.420, 0.523] | 0.472 [0.411, 0.534] | 0.469 [0.404, 0.530] |
| ADCD-Net, Full | 0.551 [0.499, 0.602] | 0.594 [0.532, 0.654] | 0.508 [0.444, 0.566] |
| DTD, Downscaled | 0.515 [0.460, 0.571] | 0.533 [0.471, 0.596] | 0.498 [0.433, 0.557] |
| DTD, Full | 0.516 [0.461, 0.570] | 0.550 [0.487, 0.611] | 0.481 [0.420, 0.549] |
| DAF (AnyText weight), Downscaled | 0.506 [0.456, 0.569] | 0.511 [0.450, 0.576] | 0.501 [0.441, 0.567] |
| DAF (AnyText weight), Full | 0.591 [0.540, 0.635] | 0.722 [0.666, 0.772] | 0.460 [0.398, 0.515] |
| DAF (DiffSTE weight), Downscaled | 0.508 [0.457, 0.561] | 0.525 [0.466, 0.584] | 0.490 [0.429, 0.553] |
| DAF (DiffSTE weight), Full | 0.608 [0.558, 0.657] | 0.718 [0.661, 0.766] | 0.498 [0.436, 0.561] |

- People minus detector AUC, with paired CIs, is in `human/auc_differences.csv`. On AI edits, people are clearly above every detector setting (smallest gap: DAF AnyText weight at full, +0.208 [+0.146, +0.273]). On handcrafted edits, the gap is clear for four of the eight settings.
- Each rater's own AUC is in `human/auc.csv` next to the mean; the two raters used the scale very differently (`human/rating_summary.csv`, `human/flag_rates.csv`).
- Rule B flags 98.9% to 100% of the 180 authentic pages for every detector setting (`human/flag_rates.csv`); people (mean score 2.5 or lower) flag 2.2% [0.5%, 4.6%].
- **Shaky rows:** any row with `small_n = True` or `shaky = yes` rests on fewer than 30 pages. The detector AUCs on the 10 AI edits both raters gave a 5 (`human/detector_auc_by_people.csv`) are shaky and must not be quoted as a finding, even where the value looks high.

## Ablation study (Week 7, rescored)

`analysis/ablation/results/` holds the rescored Week 7 study: 149 source pages (100 TextCtrl pages, plus 49 pages shared by the AnyText and DiffSTE sets), each edited by AI and by three recreated older forgery types in the same box, saved as PNG and JPEG, at both sizes. Its README explains the design, the headline contrasts and the caveats. In short: on the PNG copies at full size, only DAF separates the AI edit (AUC 0.688 [0.639, 0.738] AnyText weight, 0.849 [0.796, 0.899] DiffSTE weight); on the JPEG copies at full size DTD also does (0.683 [0.634, 0.731]), which may come from DTD's quality-100 re-save of PNG pages rather than the edit.

## Reading rules (used in the write-up, slides and project page)

- Lead with AUC. Report rule B only with its false-alarm rate, and rule A only as a contrast.
- Never quote location recall without the false-region count.
- Every number with its 95% CI and its n. Write "suggests", never "proves". Do not describe an AUC as "chance"; say "close to 0.5" or "the interval includes 0.5".
- Show the two DAF weights separately; never pool them.
- Do not quote shaky rows.
