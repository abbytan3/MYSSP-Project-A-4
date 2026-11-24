# prior_work/abby/: Weeks 4 to 7 code (as run, not re-run)

Abby's notebooks from the earlier weeks, kept as they were run so the earlier method choices can be checked. All of them ran on **Kaggle** (GPU T4 x2 where a GPU was needed), not on Modal, and wrote PNG result maps and CSV tables, not the `.npz` format of the final study. Outputs are cleared and real paths are replaced by `PATH_TO_*` placeholders (NDA). Nothing here was re-run for the submission.

## week4_textctrl/: making and grading AI edits with TextCtrl

TextCtrl at commit `ee08698` (https://github.com/weichaozeng/TextCtrl), seed 42. The TextCtrl code itself is not included; the notebooks clone it.

| Notebook | What it did |
|---|---|
| `textctrl-app-setup.ipynb` | Set up Kaggle for TextCtrl (clone, packages, Stable Diffusion weights) and ran its smoke test on the repo's example images |
| `textctrl-app-dataset.ipynb` | Picked pages, found words with OCR, planned the edits, ran a timed pilot and the full run (about 100 edited pages), pasted the edits back and wrote masks and `results.csv` |
| `textctrl-app-grading-forgeries.ipynb` | Graded the edits: OCR accuracy, NED, SSIM and PSNR, a manual blind test (asks for typed input), the worst failures, and whether ELA and SRM show the edit |

These TextCtrl pages were the AI-edit set of the Week 5 run and of the Week 7 TextCtrl set. The final study's AI-edited pages come from the mentor's dataset instead (AnyText, UDiffText, TextCtrl).

## week5_adcd/: first ADCD-Net run

| Notebook | What it did |
|---|---|
| `adcd-net-setup-inference.ipynb` | Set up ADCD-Net on Kaggle (Python 3.10, torch 2.3.0+cu121), loaded the released weights, ran pre-flight checks and saved one probability map per TextCtrl page. It also saved the quantisation table `qt_table.pk` that the final ADCD-Net job reuses |
| `adcd-net-evaluate.ipynb` | Scored those maps against the edit masks with confidence intervals (headline metrics, breakdowns, size trend) |

This is the "earlier version of our pipeline" in which ADCD-Net was checked (write-up, Limitations). The setup check in it compared against a value from a later benchmark, not from the ADCD-Net paper, and its outputs are cleared, so it is not quoted as a result.

## week7/: the ablation study (Week 7 v2)

Design: 149 source pages (100 TextCtrl pages; 49 pages shared by the AnyText and DiffSTE sets). On each page the same box was edited five ways (AI edit, generation, copy-move, splicing, untouched), each saved as PNG and as JPEG (quality 90), and every image was run at two sizes.

| File | What it did |
|---|---|
| `w7v2_01_make_sets.ipynb` | Built the page sets: the five versions in two formats, one box mask per page, a manifest and quality checks; zipped as a Kaggle dataset |
| `w7v2_02A_adcd.ipynb` | Ran ADCD-Net on every image, downscaled (`shrink512`) and full size in 512 × 512 pieces, writing one 8-bit result map per image and size |
| `w7v2_02B_dtd.ipynb` | Ran DTD the same way, in its own Python environment (torch 1.13.1+cu117) |
| `w7v2_03_evaluate.ipynb` | Evaluated every detector found (ADCD-Net, DTD, each DAF weight) on every set: per-page and per-location tables (`per_page.csv`), figures and a hand-off pack |
| `make_fake_test.py` | Wrote a tiny fake input in the same layout (random maps), so notebook 3 could be tested without real data |

The DAF results came from Nazhan's notebook (`prior_work/nazhan/`). The per-page tables from notebook 3 are what `analysis/ablation/` rescores with the final scoring rules; the Week 7 tables and figures themselves are not quoted in the submission.

## Running this code

1. Replace each `PATH_TO_*` placeholder with your own dataset or folder (for example `PATH_TO_KAGGLE_USERNAME`, `PATH_TO_DATA500_DATASET`, `PATH_TO_W7_COMMON_SETS_DATASET`, `PATH_TO_ADCD_WEIGHTS_DATASET`, `PATH_TO_TEXTCTRL_REPO`).
2. Follow the "Before you run anything" cell at the top of each notebook (Kaggle accelerator, internet and inputs). Package versions are in each notebook's install cells.
3. Run cells in order. Some helper scripts from that period were not kept, so a notebook may need small fixes to run.
