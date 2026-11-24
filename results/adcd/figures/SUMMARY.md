# ADCD-Net analysis figures (T16)

What each figure shows. Numbers come from `results/adcd/scores.csv` and `results/adcd/differences.csv` (scored by `shared/score.py`). Brackets and "x to y" pairs are 95% confidence intervals (CIs): 1,000 bootstrap resamples of documents, seed 42. This file says what the figures show, not why. All pages: 1,006 edited pages (1,093 edits) and 644 untouched pages, at both sizes. Drawn by `analysis/figures/make_figures_t16.py`.

**Figure 1 (`fig1_auc.png`): AUC by input size.** At shrink512 the AUC is 0.503 (0.486 to 0.522). At full it is 0.533 (0.510 to 0.557). Both are close to 0.5. The shrink512 CI includes 0.5. The full CI does not. The difference shrink512 minus full is -0.029 (-0.059 to -0.002).

**Figure 2 (`fig2_rule_b.png`): rule B recall vs false-alarm rate.** At shrink512, rule B recall is 0.998 (0.995 to 1.000) and the false-alarm rate is 0.995 (0.991 to 1.000). At full, recall is 0.983 (0.975 to 0.991) and the false-alarm rate is 0.986 (0.975 to 0.994). Both points lie at the top right of the plot, next to the no-skill line. Rule A is drawn at shrink512 only, for contrast with rule B at that size: recall 0.707 (0.668 to 0.744), false-alarm rate 0.693 (0.655 to 0.727). Rule A is not drawn at full size and is not used to compare sizes.

**Figure 3 (`fig3_location.png`): location recall with false regions.** At shrink512, location recall is 0.101 (0.083 to 0.120): 110 of 1,093 edits found, with 8,419 false regions. At full, it is 0.612 (0.581 to 0.645): 669 of 1,093 edits found, with 262,769 false regions. False regions are counted on all pages.

**Figure 4 (`fig4_ai_vs_hand.png`): AI-edited vs handcrafted.**
- *AUC, shrink512:* AI 0.520 (0.497 to 0.543), handcrafted 0.487 (0.465 to 0.511). AI minus handcrafted is +0.033 (+0.005 to +0.059).
- *AUC, full:* AI 0.559 (0.531 to 0.585), handcrafted 0.506 (0.477 to 0.535). Difference +0.053 (+0.025 to +0.083).
- *Location recall, shrink512:* AI 0.148 (0.120 to 0.177), 80 of 541 edits found, 5,825 false regions. Handcrafted 0.054 (0.036 to 0.077), 30 of 552 found, 6,008 false regions. Difference +0.094 (+0.060 to +0.128).
- *Location recall, full:* AI 0.573 (0.527 to 0.619), 310 of 541 found, 181,000 false regions. Handcrafted 0.650 (0.612 to 0.691), 359 of 552 found, 181,133 false regions. Difference -0.077 (-0.138 to -0.019).
- n: 506 AI-edited pages and 500 handcrafted pages, with 644 untouched pages for AUC.
- The false-region counts of the two groups each include the false regions on the 644 untouched pages, so they do not add up to the totals in figure 3. At shrink512, 5,825 + 6,008 = 11,833 against a total of 8,419 (3,414 more). At full, 181,000 + 181,133 = 362,133 against 262,769 (99,364 more). Both group rows in `scores.csv` list 644 untouched pages.

**Figure 5 (`fig5_breakdowns.png`): AUC by editor/method and by capture condition.** Each group is shown with its n of edited pages (untouched pages: 644 for every editor/method group). All values are AUC with 95% CI, shrink512 then full.
- *Editor/method:*
  - AnyText (n = 177): 0.470 (0.432 to 0.508), 0.572 (0.526 to 0.613).
  - TextCtrl (125): 0.522 (0.474 to 0.573), 0.515 (0.469 to 0.560).
  - UDiffText (204): 0.561 (0.522 to 0.596), 0.575 (0.533 to 0.617).
  - retypeset (333): 0.486 (0.456 to 0.518), 0.506 (0.473 to 0.538).
  - copymove (95): 0.475 (0.421 to 0.529), 0.491 (0.436 to 0.553).
  - pdf_edit (72): 0.508 (0.452 to 0.569), 0.528 (0.461 to 0.595).
- *Capture condition* (n edited / n untouched):
  - native (181 / 115): 0.497 (0.438 to 0.559), 0.575 (0.516 to 0.638).
  - jpeg (409 / 249): 0.491 (0.447 to 0.532), 0.511 (0.468 to 0.554).
  - resize_jpeg (205 / 150): 0.544 (0.484 to 0.603), 0.481 (0.417 to 0.542).
  - scan_sim (211 / 130): 0.493 (0.434 to 0.556), 0.574 (0.515 to 0.632).
- CIs that exclude 0.5: UDiffText at both sizes; AnyText, native and scan_sim at full only. All other group CIs include 0.5. Every AUC in the figure is close to 0.5; AnyText at shrink512 (0.470) and resize_jpeg at full (0.481) are below 0.5. The groups are shown side by side and are not compared with each other.

## Read with care

- **Rule B flags nearly every page.** The false-alarm rate is 0.995 at shrink512 and 0.986 at full, almost the same as recall. Rule B recall alone says little.
- **The AI vs handcrafted location difference flips sign between sizes.** It is +0.094 (+0.060 to +0.128) at shrink512 and -0.077 (-0.138 to -0.019) at full. The AUC difference is positive at both sizes.
- **Location recall comes with very large false-region counts** (8,419 at shrink512, 262,769 at full). Never quote recall without them.
- **Groups with only 4 to 6 found edits have shaky CIs.** At shrink512: copymove 4 of 100 edits found (0.040, CI 0.009 to 0.082), pdf_edit 5 of 72 (0.069, 0.016 to 0.133), TextCtrl 6 of 129 (0.047, 0.015 to 0.085). These location values are in `scores.csv`; they are not drawn in figure 5, which shows AUC only.
- **Group CIs are wide.** Most editor/method and capture CIs are about 0.07 to 0.2 wide and overlap each other. No pairwise group comparisons are made (`differences.csv` has none for editor or capture).
- **Rule A is not used to compare sizes.** It is shown only at shrink512, for contrast with rule B at that size. At full size rule A flags every page (recall 1.000 and false-alarm rate 1.000 in `scores.csv`), which is why it is not drawn there.
- **AUC axes are zoomed** (0.4 to 0.7 in figures 1 and 4, 0.38 to 0.86 in figure 5) so the CIs can be read. The axis labels say so.
- **Chart labels.** These PNGs were drawn during the analysis, and their legends use older wording for the AUC = 0.5 line and call the intervals "95% ranges". The write-up's figures were redrawn from the same CSV files with the final wording ("AUC = 0.5", "95% CI"); they are in `results/writeup/figures/`, drawn by `analysis/figures/make_writeup_figures.py`. The numbers are the same.
