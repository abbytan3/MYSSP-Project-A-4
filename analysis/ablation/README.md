# Week 7 ablation (T20): document-level rules A and B, AUC and "edit found"

Run (needs numpy and scipy; `shared/score.py` from T11 supplies rule A, rule B, AUC and the bootstrap). From the repo root, with `W7` set to the Week 7 folder on Abby's Mac (outside the repo; the input files are never committed):
```
W7="<Week 7 folder>/week7_safe/eval_outputs_safe"
python3 analysis/ablation/score_ablation.py "$W7/adcd/per_page.csv" "$W7/dtd/per_page.csv" "$W7/daf_anytext/per_page.csv" "$W7/daf_diffste/per_page.csv"
```
Add `--test` for a quick run on 12 pages per set. The script skips any `weights == srnet` rows, so passing `daf_srnet/per_page.csv` too makes no difference. Two runs give a byte-identical output file.
Output: `analysis/ablation/results/ablation_scores.csv`, one row per detector, weights, size, format, set and edit type, with each metric and its 95% range.

- **Rule B and AUC come from `shared/score.py` (T11, commit 2abe755).** The T11 real-page check of that file has passed (commit f0a4dc8) and the file is unchanged since 2abe755, so the ablation needs no rerun. Rerun it if `shared/score.py` changes.
- **Not included: false-region counts.** The Week 7 heat maps were not kept, so regions that overlap no box cannot be recomputed. Location precision and location F1 are therefore not reported here; only "edit found" (`loc_recall`) is.
- Rule A: a page is "tampered" if any pixel is flagged (`flagged_any`). Rule B: tampered if the page's top score (`max_value`) passes a cut-off chosen by 2-fold cross-fitting on the other half of the documents (the cut-offs used are in the last two columns). AUC comes from the top scores. Clean pages for each cell are the `untouched` rows of the same detector, weights, size, format and set.
- Edit found = the file's `detected` column (1 when a region overlaps the edit box; equals recall > 0 on every tampered row).
- **Coverage:** four detector/weight combinations from the `eval_outputs_safe/*/per_page.csv` files: ADCD-Net, DTD, DAF (AnyText weight) and DAF (DiffSTE weight). The DAF SRNet file is excluded by Abby's decision, to match the main study's 2 DAF weights (PLAN.md section 3: "SRNet weight not used"), although Week 7 ran 3 weights. Limitation: Week 7 figures that used the SRNet weight can't be reproduced from this table. All five files hold the same 198 (set, page ID) pairs, so pages are paired across detectors. DAF also has a third size, `default`, besides `shrink512` and `full`; it is reported as is.
- The `untouched` pages are shared across versions, so cells are not independent.
- Check against Week 5: ADCD-Net, TextCtrl, 512, PNG, AI edit gives edit found 5/100 and 84/100 untouched pages flagged by rule A. Both match.
- **Page counts match the Week 7 design.** Each file holds 939 images per size and format (TextCtrl 485, AnyText 227, DiffSTE 227), 1,878 rows per size over PNG and JPG. The AnyText and DiffSTE sets share the same 49 source pages, and their untouched, generation, splicing and copy-move versions are the same images listed under both sets: 49 + 49 + 45 + 35 = 178 per format. Counting them once gives 939 - 178 = 761 images, and 761 x 2 formats = 1,522, the design's total. Checked in all four files: every one of these rows has identical `max_value` and `flagged_any` in both sets (0 differ). These 178 images per format are not independent across the two sets: never pool their untouched or older-edit rows across AnyText and DiffSTE. Only the `ai_edit` rows differ between the two sets. For any pooled result, group by source page across the two sets.
- Documents: the page ID is the source page shared by all 5 versions and both formats, and AnyText and DiffSTE share their `imgNN` pages, so one page ID is one document (assumed from the file, which has no document ID). The 2-fold split and the bootstrap (1,000 resamples, seed 42) use these documents, so a page's rows always stay together and the 49 shared pages are never counted twice.
- Page IDs in the input are `page_NNNN` (TextCtrl) and `imgNN` (AnyText, DiffSTE); both are anonymous.

## How to read the table

- **Rule B flags every page in many cells.** In most cells the tampered and untouched groups are about equal in size, so flagging every page gives precision 0.5, recall 1.0 and F1 near 0.667, which looks decent but means nothing. Read rule B only together with its false-alarm rate and the AUC. The main run (W1) found the same thing.
- Rule A and rule B ranges collapse to a single value where a metric is exactly 0 or 1 in every resample (for example, flagging every page); that is a saturated result, not a missing range.
