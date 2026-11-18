# Week 7 ablation (T20), part 1: rule A and "edit found"

Run: `python3 analysis/ablation/score_ablation.py PER_PAGE_CSV` (add `--test` for 3 pages per set).
Output: `analysis/ablation/results/ablation_rule_a_found.csv` (one row per detector, size, format, set, version).

- **Not included: false-region counts.** The Week 7 heat maps were not kept, so regions that overlap no box cannot be recomputed. Location precision and location F1 are therefore not reported here; only "edit found" (location recall) is.
- **Not included yet: rule B** (cross-fitted cut-off). It needs the shared scoring functions from T11; this part will be added after they are pushed to `main`.
- Rule A: a page is "tampered" if any pixel is flagged (`flagged_any`). Clean pages are the `untouched` rows of the same detector, size, format and set.
- Edit found = the file's `detected` column (1 when a region overlaps the edit box; checked: it equals recall > 0 on every tampered row).
- **Coverage:** four detector/weight combinations from the five `eval_outputs_safe/*/per_page.csv` files: ADCD-Net, DTD, DAF (AnyText weight) and DAF (DiffSTE weight). The DAF SRNet file is skipped because the plan does not use that weight. DAF also has a third size, `default`, besides `shrink512` and `full`; it is reported as is.
- Ranges: 95% from 1,000 bootstrap resamples, seed 42. The unit is the source page (set + page ID): one draw takes all of that page's rows (every version, format, size and detector) together, never single rows. If a page ID is the same source page for all 5 versions and both formats, this is resampling by document. The file has no document ID, so that is assumed, not checked.
- The `untouched` pages are shared across versions, so cells are not independent.
- Page IDs are `page_NNNN` (TextCtrl) and `imgNN` (AnyText, DiffSTE); both are anonymous, and are keyed with the set name so they never clash.
- Check against Week 5: ADCD-Net, TextCtrl, 512, PNG, AI edit gives edit found 5/100 and 84/100 untouched pages flagged by rule A. Both match.
