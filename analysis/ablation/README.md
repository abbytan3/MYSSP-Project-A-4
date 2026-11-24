# Week 7 ablation study (T20): rules A and B, AUC and "edit found"

The ablation study rescores the Week 7 study with the final study's scoring rules (`shared/score.py`), so both are read the same way. No GPU is needed.

## What the Week 7 study was

- **149 source pages:** 100 TextCtrl pages, and 49 pages shared by the AnyText and DiffSTE sets (same pages, same boxes, same new text).
- On each page the same box was changed five ways: **AI edit**, three recreated older forgery types (**generation**, **copy-move**, **splicing**), and **untouched**. In the TextCtrl set, copy-move exists for 86 pages and splicing for 99 (every count is in the files).
- Every image was saved as **PNG** and as **JPEG** (quality 90), and run by ADCD-Net, DTD and DAF (AnyText and DiffSTE weights) at two sizes, downscaled (`shrink512`) and full. DAF also has a third Week 7 size, `default` (the page as it is; DAF resizes it itself), reported as is.
- The notebooks that built and ran it are in `prior_work/abby/week7/` and `prior_work/nazhan/week7/`.

## Run

Needs numpy and scipy. From the repo root, with `W7` set to the folder that holds the Week 7 per-page tables (private, outside the repo; the input files are never committed):

```
W7="<Week 7 folder>/week7_safe/eval_outputs_safe"
python3 analysis/ablation/score_ablation.py "$W7/adcd/per_page.csv" "$W7/dtd/per_page.csv" "$W7/daf_anytext/per_page.csv" "$W7/daf_diffste/per_page.csv"
python3 analysis/ablation/make_contrast_summary.py "$W7/adcd/per_page.csv" "$W7/dtd/per_page.csv" "$W7/daf_anytext/per_page.csv" "$W7/daf_diffste/per_page.csv"
```

Add `--test` to `score_ablation.py` for a quick run on 12 pages per set. Rows with `weights == srnet` are skipped, so passing `daf_srnet/per_page.csv` too makes no difference. Two runs give byte-identical output files.

## Outputs

- `results/ablation_scores.csv`: 240 rows, one per detector, weights, size, format, set and edit type, each metric with its 95% confidence interval (`_lo`, `_hi`) and counts. The rule B cut-offs used are in the last two columns.
- `results/contrast_summary.csv`: 148 paired contrasts (TextCtrl set only), each with the AUC and "edit found" on both sides, the difference with its 95% CI, and `clear` (the CI excludes 0).

## Scoring

- **Rule A:** a page is "tampered" if any pixel is flagged (`flagged_any`). **Rule B:** tampered if the page's top score (`max_value`) passes a cut-off chosen by 2-fold cross-fitting on the other half of the documents. **AUC** from the top scores. All three come from `shared/score.py` (unchanged since commit 2abe755, whose real-page check passed in f0a4dc8), imported, not copied.
- **Edit found** = the file's `detected` column (1 when a region overlaps the edit box).
- **Clean pages** for each cell are the `untouched` rows of the same detector, weights, size, format and set.
- **Documents:** one page ID is one document (assumed from the IDs; the files have no document ID). The 2-fold split and the bootstrap (1,000 resamples, seed 42) keep all rows of a page together.
- **Not included:** false-region counts, location precision and location F1. The Week 7 heat maps were not kept, so regions that overlap no box cannot be recomputed.
- **SRNet:** DAF's SRNet weight ran in Week 7 but is left out, to match the main study's two DAF weights. Week 7 figures that used it can't be reproduced from these tables.

## Headline: AI edit vs recreations, TextCtrl set, PNG copies

AUC with 95% CI (contrast A in `contrast_summary.csv`; a star marks a difference whose CI excludes 0). AI edit side: 100 edited and 100 untouched pages.

| Detector | Size | AUC, AI edit | AI minus generation | AI minus copy-move | AI minus splicing |
|---|---|---|---|---|---|
| ADCD-Net | Downscaled | 0.51 | +0.00 (-0.00 to +0.01) | +0.00 (-0.02 to +0.03) | -0.00 (-0.02 to +0.01) |
| ADCD-Net | Full | 0.53 | +0.03 (+0.02 to +0.06) * | +0.05 (+0.02 to +0.09) * | +0.03 (+0.01 to +0.06) * |
| DTD | Downscaled | 0.52 | +0.00 (-0.02 to +0.02) | +0.00 (-0.03 to +0.03) | +0.02 (+0.00 to +0.03) * |
| DTD | Full | 0.50 | +0.01 (+0.00 to +0.02) | +0.02 (-0.00 to +0.04) | +0.01 (+0.00 to +0.02) |
| DAF, AnyText weight | Downscaled | 0.51 | +0.01 (-0.00 to +0.02) | +0.00 (-0.03 to +0.03) | +0.01 (-0.00 to +0.03) |
| DAF, AnyText weight | Full | 0.69 | +0.17 (+0.11 to +0.23) * | +0.19 (+0.13 to +0.25) * | +0.15 (+0.09 to +0.21) * |
| DAF, DiffSTE weight | Downscaled | 0.51 | +0.00 (-0.01 to +0.02) | +0.02 (-0.02 to +0.05) | -0.01 (-0.04 to +0.00) |
| DAF, DiffSTE weight | Full | 0.85 | +0.32 (+0.26 to +0.38) * | +0.35 (+0.29 to +0.41) * | +0.31 (+0.26 to +0.37) * |

Exact AI-edit AUCs at full size, PNG (`ablation_scores.csv`): DAF AnyText weight 0.688 [0.639, 0.738], DAF DiffSTE weight 0.849 [0.796, 0.899], ADCD-Net 0.527 [0.511, 0.549], DTD 0.501 [0.489, 0.513].

Other contrasts in the file: **B**, full minus downscaled (and DAF `default` minus downscaled as its own rows); **C**, PNG minus JPEG. Clear AUC differences: A 28 of 60, B full minus downscaled 10 of 32, B default minus downscaled 4 of 16, C 13 of 40. CIs in `contrast_summary.csv` differ slightly from `ablation_scores.csv` because the bootstrap draws are not the same; the AUC point estimates are identical.

## Read with care

- **The headline holds for the PNG copies only.** On the JPEG copies at full size, DTD also separates the AI edit: AUC 0.683 [0.634, 0.731] (DAF: 0.741 AnyText weight, 0.830 DiffSTE weight). DTD reads only JPEG, and PNG pages are re-saved at quality 100 before it reads them, so the PNG/JPEG difference may come from the re-save rather than the edit. The cause was not tested.
- **Rule B flags every page in many cells.** With about equal tampered and untouched groups, flagging everything gives precision 0.5, recall 1.0 and F1 near 0.667, which means nothing. Read rule B only with its false-alarm rate and the AUC.
- **One small set.** The contrasts are paired comparisons within the TextCtrl set; the recreations are made from the same pages.
- **The AnyText and DiffSTE sets share 178 images per format** (untouched, generation, splicing and copy-move on the same 49 pages; 49 + 49 + 45 + 35). They appear with identical scores under both sets; only the `ai_edit` rows differ. Never pool their rows across the two sets; group by source page instead. This also explains the image count: 939 per size and format in each file, 761 distinct, 1,522 over both formats.
- **Untouched pages are shared across versions**, so cells within a set are not independent.
- CIs collapse to a single value where a metric is exactly 0 or 1 in every resample (for example, flagging every page); that is a saturated result, not a missing CI.
- **Check against Week 5:** ADCD-Net, TextCtrl, downscaled, PNG, AI edit gives edit found 5 of 100 and rule A flags 84 of 100 untouched pages, matching the Week 5 run.
- Page IDs in the input are `page_NNNN` (TextCtrl) and `imgNN` (AnyText, DiffSTE); both are anonymous.
