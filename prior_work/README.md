# prior_work/: earlier-week code (kept as run, not re-run)

Code from each team member's earlier weeks (Weeks 3 to 7), kept for reproducibility and audit. It shows how the methods of the final study were first worked out. None of it was re-run for the submission, and nothing in the main pipeline (`shared/`, `detectors/`, `analysis/`) imports it.

Before it was committed, every notebook's outputs were cleared and real paths, dataset names and usernames were replaced with `PATH_TO_*` (or similar) placeholders, because the pages are under NDA.

| Folder | Whose | What it holds | Which part of the submission it relates to |
|---|---|---|---|
| [`abby/`](abby/README.md) | Abby | Week 4 TextCtrl editing and grading, Week 5 ADCD-Net first run and evaluation, Week 7 ablation study notebooks (build the page sets, run ADCD-Net and DTD, evaluate) | Week 7 notebooks produced the per-page tables that `analysis/ablation/` rescores; Week 5 is the earlier ADCD-Net check mentioned in the write-up's limitations |
| [`alycia/`](alycia/README.md) | Alycia | Weeks 3 to 6 AnyText editing code (`make_tampered.py`) and a Colab test notebook | Background for how AnyText edits and their masks were made |
| [`nazhan/`](nazhan/README.md) | Nazhan | Week 7 DAF notebook (three weights, three sizes) | The DAF rows of the ablation study, and where the final DAF job's settings were worked out |

## What is not here

- The final study's code: it is in `shared/`, `detectors/` and `analysis/`.
- Third-party code (TextCtrl, AnyText, the detectors), weights, pages, masks, result maps and per-page CSVs with real names.
- A few small helper scripts from that period that were not kept (for example the Week 7 tiling helper; `detectors/adcd/tiling.py` rebuilds it). Some notebooks may therefore not run as they stand.
- No Week 7 DTD notebook from Alycia was found; the Week 7 DTD run is Abby's `abby/week7/w7v2_02B_dtd.ipynb`.
