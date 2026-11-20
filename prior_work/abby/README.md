# prior_work/abby/: Weeks 4–7 code (as run, not re-run)

This folder holds Abby's code from the earlier weeks of MYSSP A4, preserved for reproducibility. All code was run exactly as here; nothing has been re-run.

## Folder structure

### week7/
**Week 7 ablation studies.** Rescores the Week 7 controlled study from safe CSV outputs (no GPU).

- **w7v2_01_make_sets.ipynb** – Built the shared test set: creates manifest, masks, and tiny test images for validating the evaluation pipeline. Used by notebooks 02A/02B for input format checking and feeds the shared dataset format to `analysis/ablation/`.

- **w7v2_02A_adcd.ipynb** – Runs ADCD-Net (at 512×512 and full size) on the shared test set; outputs per-page heat maps to Modal Volume. Feeds Section 3 (detector runs) of the PLAN.

- **w7v2_02B_dtd.ipynb** – Runs DTD (at 512×512 and full size) on the shared test set; outputs per-page heat maps to Modal Volume. Feeds Section 3 (detector runs) of the PLAN.

- **make_fake_test.py** – Generates 4 fake pages per set for testing the evaluation notebook without real data. Checks the output format and timing before the full run. Requires `PATH_TO_REPO/docs/decisions.md`.

- **w7v2_03_evaluate.ipynb** – The shared scoring script: loads Modal outputs (.npz heat maps); scores per-page and per-location; builds result tables and figures for `analysis/ablation/`. Feeds Section 4 (scoring) of the PLAN. Requires `PATH_TO_SAMPLE_MAP_CSV` for page metadata.

### week5_adcd/
**Week 5 ADCD-Net validation on AI edits.** Tests ADCD-Net on TextCtrl-generated pages; cited in the write-up Methodology section.

- **adcd-net-setup-inference.ipynb** – Sets up ADCD-Net on Kaggle: environment, packages, pre-trained weights, pre-flight checks, inference on TextCtrl pages. Documents the validation of ADCD-Net's performance on AI-edited bank statements.

- **adcd-net-evaluate.ipynb** – Evaluates ADCD-Net's output: computes scores on TextCtrl pages, sanity checks (T-SROIE validation), and performance tables. Outputs feed the write-up's Methodology section.

### week4_textctrl/
**Week 4 TextCtrl setup and execution.** Code to generate and grade AI-edited pages with TextCtrl.

- **textctrl-app-setup.ipynb** – Documents TextCtrl library setup on Kaggle: download from GitHub (commit `ee08698`, https://github.com/weichaozeng/TextCtrl.git), install packages, download Stable Diffusion v1.5 weights, pre-flight checks. Provides reproducible setup trace.

- **textctrl-app-dataset.ipynb** – Runs TextCtrl to generate AI edits on bank statements. Covers data inspection, word selection, pilot run, and full generation. Outputs are the AI-edited pages that feed into the detector runs (Section 1–2 of the PLAN). Seed = 42.

- **textctrl-app-grading-forgeries.ipynb** – Grades TextCtrl output: measures quality via OCR accuracy, NED, SSIM, PSNR; includes a manual blind test for visual assessment; detects which edits are visible to classical forensic tools (ELA, SRM). Grading is **scripted with manual interactive blind test** (Steps 5–6 ask for typed input).

## TextCtrl library not included

The TextCtrl library code itself is not in this folder; the notebooks reference it via `PATH_TO_TEXTCTRL_REPO`. The notebooks were run against TextCtrl commit `ee08698` from https://github.com/weichaozeng/TextCtrl.git.

## Shared utilities

The Python files referenced by multiple notebooks (e.g. `make_versions.py`, `cut_and_join.py`, `score_maps.py` for building the test set) are either defined inline as notebook cells or are part of the TextCtrl library code, not stored separately.

## Path replacements for public repository

Real paths have been replaced with placeholders (all prefixed `PATH_TO_`) to comply with NDA:

- `PATH_TO_KAGGLE_USERNAME` – Kaggle workspace username
- `PATH_TO_DATA500_DATASET` – 500-page sample dataset
- `PATH_TO_W7_COMMON_SETS_DATASET` – Week 7 test set (manifest, masks, images)
- `PATH_TO_ADCD_WEIGHTS_DATASET`, etc. – Model weights datasets
- `PATH_TO_STAGE1_SPLIT` – Real bank statement pages and labels (outside repo)
- `PATH_TO_SAMPLE_MAP_CSV` – Anonymous page ID mapping (never committed)
- `PATH_TO_REPO/docs/decisions.md` – Project decisions log

## Running this code

These notebooks were run exactly as shown here on Kaggle (GPU T4 x2, Python 3.13, torch 2.11). They are preserved as-is for reproducibility and are **not re-run** as part of the submission's pipeline. To use them:

1. Replace each `PATH_TO_*` placeholder with the actual path or dataset name in your environment.
2. Follow the "Before you run anything" instructions in each notebook's first cell (Accelerator, Internet, Add Input settings for Kaggle).
3. Run the notebook cells sequentially; do not skip cells or run non-interactively unless a cell explicitly says so.

---

**Author:** Abby Tan  
**Notebooks last run:** Weeks 4–7, 2026  
**TextCtrl version:** commit ee08698 from https://github.com/weichaozeng/TextCtrl.git
