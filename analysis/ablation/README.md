# Week 7 ablation (T20), part 1: rule A and "edit found"

Run: `python3 analysis/ablation/score_ablation.py PER_PAGE_CSV` (add `--test` for 3 pages per set).
Output: `analysis/ablation/results/ablation_rule_a_found.csv` (one row per detector, size, format, set, version).

- **Not included: false-region counts.** The Week 7 heat maps were not kept, so regions that overlap no box cannot be recomputed. Location precision and location F1 are therefore not reported here; only "edit found" (location recall) is.
- **Not included yet: rule B** (cross-fitted cut-off). It needs the shared scoring functions from T11; this part will be added after they are pushed to `main`.
- Rule A: a page is "tampered" if any pixel is flagged (`flagged_any`). Clean pages are the `untouched` rows of the same detector, size, format and set.
- Edit found = the file's `detected` column (1 when a region overlaps the edit box; checked: it equals recall > 0 on every tampered row).
- Ranges: 95% from 1,000 bootstrap resamples of pages, seed 42. The safe file has no document ID, so a page is the unit; this may make ranges slightly too narrow where several pages share a document.
- The input only holds ADCD-Net runs. The `untouched` pages are shared across versions, so cells are not independent.
- Page IDs in the input are `page_NNNN` (TextCtrl) and `imgNN` (AnyText, DiffSTE); both are anonymous, and are keyed together with the set name so they never clash.
