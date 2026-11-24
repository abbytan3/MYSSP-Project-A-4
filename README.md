# Can Off-the-Shelf Tampering Detectors Find AI Edits in Bank Statements?

**MYSSP 2026, Project A-4** · Abby Tan Yi Pei, Alycia Yeak Yinyi, Muhammad Nazhan bin Zailani · Mentor and co-author: Jia Yu Lim

- **Project page** (write-up, video and Demo Day slides): https://myssp.myresearchguide.org/#/project-archive/project/financial-document-tampering

## What this project does

We test whether three pre-trained document-tampering detectors (ADCD-Net, DTD and DAF), used exactly as released with no fine-tuning, can tell AI-edited bank statements from untouched ones and point to the edited field. AI edits (AnyText, UDiffText, TextCtrl) are compared with handcrafted edits (retyped text, copy-move, PDF edits) on 1,650 real Malaysian bank-statement pages, and with what two human raters saw by eye.

**Research questions**

- **RQ1, detection:** how well do the detectors tell edited statements from their untouched originals, and how often do they flag an untouched statement?
- **RQ2, localisation:** how often do they point to the field that was actually edited, and how many regions do they flag where nothing changed?
- **RQ3, conditions:** how do the AI editor, the page's capture condition, the input size and how convincing people found the edit change their performance?

The confirmed plan is in [`docs/PLAN.md`](docs/PLAN.md) (frozen on 18 Nov 2026). Every method choice and every difference from the plan is logged, with its reason, in [`docs/decisions.md`](docs/decisions.md).

## Results at a glance

All numbers are copied from the committed CSV files and rounded to 3 decimals; brackets hold 95% confidence intervals from 1,000 bootstrap resamples of documents. A guide to every results file, with the breakdowns, is in [`results/README.md`](results/README.md).

**RQ1, detection** (1,006 edited pages vs 644 untouched pages; source `results/<detector>/scores.csv`). AUC is 0.5 when a detector's scores do not separate edited from untouched pages and 1 when they separate them perfectly.

| Detector | Input | AUC, all edited | AUC, AI-edited (506) | AUC, handcrafted (500) | Rule B false-alarm rate |
|---|---|---|---|---|---|
| ADCD-Net | Downscaled | 0.503 [0.486, 0.522] | 0.520 [0.497, 0.543] | 0.487 [0.465, 0.511] | 0.995 [0.991, 1.000] |
| ADCD-Net | Full | 0.533 [0.510, 0.557] | 0.559 [0.531, 0.585] | 0.506 [0.477, 0.535] | 0.986 [0.975, 0.994] |
| DTD | Downscaled | 0.507 [0.482, 0.532] | 0.516 [0.486, 0.544] | 0.498 [0.470, 0.525] | 0.995 [0.989, 1.000] |
| DTD | Full | 0.505 [0.478, 0.532] | 0.535 [0.505, 0.564] | 0.475 [0.444, 0.506] | 0.997 [0.992, 1.000] |
| DAF (AnyText weight) | Downscaled | 0.505 [0.487, 0.522] | 0.509 [0.486, 0.533] | 0.501 [0.481, 0.521] | 0.995 [0.989, 1.000] |
| DAF (AnyText weight) | Full | 0.598 [0.577, 0.622] | 0.710 [0.682, 0.738] | 0.484 [0.459, 0.508] | 0.991 [0.983, 0.997] |
| DAF (DiffSTE weight) | Downscaled | 0.512 [0.492, 0.530] | 0.515 [0.490, 0.540] | 0.509 [0.484, 0.530] | 1.000 [1.000, 1.000] |
| DAF (DiffSTE weight) | Full | 0.595 [0.574, 0.618] | 0.698 [0.673, 0.727] | 0.490 [0.464, 0.518] | 1.000 [1.000, 1.000] |

**RQ2, localisation:** detectors find edits only by flagging many unchanged regions. For example, ADCD-Net at full size found 669 of 1,093 edits (location recall 0.612 [0.581, 0.645]) while flagging 262,769 regions where nothing changed; the highest location precision is 0.105 [0.100, 0.110] (DTD, downscaled).

**RQ3, people vs detectors** (500 real pages rated by two team members; source `results/human/auc.csv`): people separated the 160 AI-edited pages from the 180 authentic pages with AUC 0.930 [0.900, 0.958]; the best detector reached 0.722 [0.666, 0.772] (DAF, AnyText weight, full size). On the 160 handcrafted pages people reached 0.567 [0.511, 0.622] and no detector setting exceeded 0.508.

**How to read these numbers**

- Rule B flags 98.6% to 100% of untouched pages for every detector and input size, so its recall means little on its own. Read rule B only next to its false-alarm rate, and lead with AUC.
- Never quote location recall without the false-region count next to it.
- Differences between editors or capture conditions were not computed as paired differences. Where two 95% confidence intervals do not overlap, the most we write is that this "suggests" a difference (see `results/README.md`).
- Rows marked `small_n` or `shaky` (fewer than 30 pages) are not findings.

## Data (read this first: it decides what you can reproduce)

The bank statements were supplied by our industry collaborator, **AI Lens Sdn. Bhd. (Kuala Lumpur)**, under a non-disclosure agreement (NDA). Our mentor anonymised them (logos, names and details changed) and made the tampered versions. **No page, crop, heat map, real file name or statement text is in this repo, and none can be shared.** Only score tables with anonymous page IDs (`page_0001` …) and charts made from them are published here.

So there are two levels of reproduction:

| Level | What you can redo | What you need |
|---|---|---|
| **A. From this repo alone** | Run the tests, rebuild the cross-detector tables, redraw every chart from the committed score files | Python and a `git clone` of this repo |
| **B. The full experiment** | Sample the pages, run the three detectors, score them, rerun the human-rating and ablation study analyses | The private dataset and rating files (not shareable under the NDA), a Modal account with a GPU, and the detector weights |

Level B is documented step by step below so the method can be repeated on other data, but it cannot be run without the private inputs.

## The pipeline (where each folder fits)

```
private stage 1 split (real pages, labels, reviewer decisions)
  → shared/sample.py        pick 1,650 pages (506 AI-edited, 500 handcrafted, 644 untouched), anonymous IDs
  → shared/upload_pages.py  pages + meta/sample.csv to a private Modal Volume
  → detectors/<name>/       ADCD-Net, DTD, DAF (AnyText and DiffSTE weights): downscaled and full size
  → out/<detector>/<size>/<page_id>.npz   top score + flagged pixels only (shared/output_format.py)
  → shared/check_outputs.py then shared/score.py   rule A, rule B, AUC, edit found, false regions, 95% CIs
  → results/<detector>/{scores,differences}.csv + figures
  → analysis/combined/      cross-detector tables  →  results/combined/
  → analysis/human/         people vs detectors on the 500 real stage 2 pages  →  results/human/
  → analysis/ablation/      Week 7 ablation study rescored (no GPU)  →  analysis/ablation/results/
```

1,650 pages × 2 input sizes × 4 detector weights = 13,200 detector runs. The sample is 1,006 edited pages (1,093 edits) from 644 documents plus their 644 untouched originals. (`docs/PLAN.md` still says about 655 / 1,661 / 13,300, the estimate before sampling; the built numbers are the ones above, see `docs/decisions.md`, T09.)

## Terms used in this repo

| Term | Meaning |
|---|---|
| Downscaled (`shrink512` in code and file names) | The whole page shrunk so its longest side is 512 pixels |
| Full (`full`) | The page at full size, cut into each detector's own tile size with no overlap and joined back: 512 × 512 for ADCD-Net and DTD, 1,536 × 1,024 for DAF. This tiling is our own design, not the authors' |
| Top score | A page's highest pixel score (0 to 1) |
| Rule A | A page is called edited if any pixel scores 0.5 or more (the authors' default) |
| Rule B | A page is called edited if its top score passes a cut-off chosen by 2-fold cross-fitting over documents, so no page chooses its own cut-off |
| AUC | How well the top scores rank edited pages above untouched ones: 0.5 = no separation, 1 = perfect |
| Region, false region | Touching flagged pixels form a region; a false region overlaps no edit box |
| Edit found, location recall | An edit is found if a region overlaps its box; location recall = edits found ÷ all edits |
| Capture condition | From each page's label: `native`, `jpeg`, `resize_jpeg`, `scan_sim` (simulated scan) |
| 95% CI | 2.5th and 97.5th percentiles of 1,000 bootstrap resamples of documents (pages of one document stay together), seed 42. A difference is "clear" only if its CI excludes 0 |

## Repository layout

| Folder | Contents |
|---|---|
| `shared/` | Sampling, upload, the common output format, the output check and the scoring script ([`shared/README.md`](shared/README.md)) |
| `detectors/adcd/`, `detectors/dtd/`, `detectors/daf/` | One Modal GPU job per detector, its tiling code and tests, and its own README |
| `analysis/combined/` | Cross-detector tables (`make_tables.py`) |
| `analysis/figures/` | Chart scripts: the write-up's Figures 2 to 5 (`make_writeup_figures.py`), cross-detector (`make_figures.py`), ADCD-Net (`make_figures_t16.py`), DAF (`make_figures_t18.py`) |
| `analysis/human/` | People vs detectors on the stage 2 ratings ([README](analysis/human/README.md)) |
| `analysis/ablation/` | The Week 7 ablation study rescored, with its results ([README](analysis/ablation/README.md)) |
| `analysis/tile_check/`, `analysis/dtd/` | Checks that page size is not a shortcut (printed output saved in `results/adcd/tile_check/`); spread of DTD's top scores |
| `results/` | Score tables and charts for every detector, the cross-detector tables and the people-vs-detectors tables ([guide](results/README.md)) |
| `prior_work/` | Earlier-week code (Weeks 3 to 7), kept as it was run, **not re-run** ([README](prior_work/README.md)) |
| `docs/` | Plan (frozen), setup guide, decisions log |
| `scripts/`, `.githooks/` | The NDA check that runs before every commit |
| `AGENTS.md`, `CLAUDE.md`, `.claude/`, `.superset/` | Rules for the coding agents and the Paradev workspace setup used during the project |

## Requirements

- Python 3.10 or newer. Pinned packages: `shared/requirements.txt` (numpy 1.26.4, scipy 1.13.1, modal 1.6.4) and `analysis/figures/requirements.txt` (matplotlib 3.9.4); `pytest` runs the tests. The GPU images install their own packages: see `detectors/*/requirements.txt`.
- Level A was also checked on Python 3.13 with numpy 2.5.3, scipy 1.18.1 and matplotlib 3.11.2 (re-checked on 24 Nov 2026): all 49 tests pass and `make_tables.py` rewrites the committed tables byte for byte. Charts redraw with the same numbers, but PNG bytes can differ between matplotlib versions.
- Level B also needs a [Modal](https://modal.com) login. Setup, Git identity and the NDA hook are in [`docs/SETUP.md`](docs/SETUP.md). Never commit the Modal token.

```bash
# clone the repo (a ZIP download has no .git folder, and one safety test needs it)
git clone https://github.com/abbytan3/MYSSP-Project-A-4.git
# go into it
cd MYSSP-Project-A-4
# turn on the NDA check that runs before every commit (once per clone)
git config core.hooksPath .githooks
# install the packages the analysis scripts need
python3 -m pip install -r shared/requirements.txt -r analysis/figures/requirements.txt pytest
```

## Level A: reproduce the tables and charts from this repo

Run from the repo root. These use only the committed files, so anyone can run them.

```bash
# run the tests for the scoring, sampling, output format, people-vs-detectors and combined tables (49 tests)
python3 -m pytest shared analysis/human analysis/combined -q
# run each detector's tiling check (random arrays, no pages)
python3 -I detectors/adcd/test_tiling.py
python3 -I detectors/dtd/test_tiling.py
python3 -I detectors/daf/test_tiling.py
# rebuild the five cross-detector tables in results/combined/tables/ from results/*/scores.csv and results/human/*.csv
python3 analysis/combined/make_tables.py
# redraw the three cross-detector charts (AUC, rule B vs false alarms, location) as PNG and PDF
python3 analysis/figures/make_figures.py --out-dir results/combined/figures
# redraw the five ADCD-Net charts in results/adcd/figures/
python3 analysis/figures/make_figures_t16.py
# redraw the five DAF charts for each weight
python3 analysis/figures/make_figures_t18.py --detector daf_anytext --name "DAF (AnyText weight)"
python3 analysis/figures/make_figures_t18.py --detector daf_diffste --name "DAF (DiffSTE weight)"
# redraw the two people-vs-detectors charts in results/human/figures/
python3 analysis/human/make_figures.py
# redraw the write-up's Figures 2 to 5 and list every plotted value in results/writeup/figures/figure_data.csv
python3 analysis/figures/make_writeup_figures.py
```

`make_tables.py` copies every value as text from the score files (no rounding, no arithmetic) and prints how many cells were missing; it should say 0. Charts are drawn only from the CSV tables.

- If `test_joined_file_has_only_the_allowed_columns_and_never_goes_inside_the_repo` fails, you are probably running from a ZIP download: the test checks that the script refuses to write inside a git repo, and a ZIP has no `.git` folder. Use `git clone`, and delete the `analysis/human/should_not_exist/` folder (a one-row made-up test file) that the failed test leaves behind.
- **What this does not redo:** the scores themselves. `scores.csv` comes from the detector outputs, which come from the private pages (Level B).

## Level B: the full experiment, step by step

`<private>` means a folder **outside this repo**. Every script takes its paths as arguments; none is stored in code. Seed 42 is used for every random choice.

### 1. Pick the pages (private inputs)

Needs the stage 1 split, the stage 1 reviewers' decisions and the stage 2 rating files. Reads metadata only and never opens a page image.

```bash
# write sample.csv (anonymous IDs, goes to the Volume) and sample_map.csv (real names, stays private) into <private>
python3 shared/sample.py --stage1-split <stage1_split> --review-results <review_results_stage1> --stage2-ratings <rating files> --out-dir <private> --test
```

Drop `--test` for the real run (`--test` also writes a 3-page test sample). `--stage2-split <dir>` is optional: the 6 real authentic stage 2 pages missing from the stage 1 split are read from it. Expected: 506 AI-edited (AnyText 177, UDiffText 204, TextCtrl 125), 500 handcrafted (retypeset 333, copy-move 95, PDF edit 72), 644 untouched = 1,650 pages.

### 2. Upload to the private Volume

```bash
# send 3 test pages and meta/sample.csv to the Volume a4-data
python3 shared/upload_pages.py --sample-map <private>/sample_map.csv --sample <private>/sample.csv --test
# send all pages
python3 shared/upload_pages.py --sample-map <private>/sample_map.csv --sample <private>/sample.csv
```

`sample_map.csv` is only read locally to find each page; it is never uploaded.

### 3. Put the detector weights on the Volume

| Detector | Code (pinned, as cloned by the job) | Weights |
|---|---|---|
| ADCD-Net (ICCV 2025) | `KAHIMWONG/ACDC-Net` @ `169d04d` | `ADCDNet.pth`, `docres.pkl` (authors' release) and `qt_table.pk` (saved by our Week 5 run) → `weights/adcd/` |
| DTD (CVPR 2023) | `qcf-568/DocTamper` @ `2d53389` | `dtd_doctamper.pth`, `vph_imagenet.pt`, `swin_imagenet.pt` (authors' Drive link in their `models/Readme.md`) → `weights/dtd/` |
| DAF (from the OSTF paper, AAAI 2025) | `qcf-568/OSTF` @ `1fec955` | `anytext.pth`, `diffste.pth` (released with the paper; the SRNet weight is not used) → `weights/daf/` |

The exact `modal volume put` commands are in each detector's README. Detectors are used exactly as released: no retraining.

### 4. Run each detector on the GPU (Modal, T4)

Every detector starts with 3 real pages at both sizes (checks the format and timing), then all pages downscaled, then full size. A stopped run can be restarted: finished pages are skipped. Full details and the checks after each stage are in each README.

```bash
# ADCD-Net: OCR text masks first (CPU), then the detector at both sizes
modal run detectors/adcd/modal_job.py --stage ocr
modal run detectors/adcd/modal_job.py --stage adcd --size shrink512
modal run detectors/adcd/modal_job.py --stage adcd --size full
# ADCD-Net: check the outputs of each stage (OCR masks, then each size)
modal run detectors/adcd/modal_job.py --stage check_ocr
modal run detectors/adcd/modal_job.py --stage check_adcd --size shrink512
modal run detectors/adcd/modal_job.py --stage check_adcd --size full
# DTD: environment check, then both sizes, then the output check for each size
modal run detectors/dtd/modal_job.py --stage env
modal run detectors/dtd/modal_job.py --stage dtd --size shrink512
modal run detectors/dtd/modal_job.py --stage dtd --size full
modal run detectors/dtd/modal_job.py --stage check_dtd --size shrink512
modal run detectors/dtd/modal_job.py --stage check_dtd --size full
# DAF: environment check, then each weight at both sizes, then the output check for every weight and size
modal run detectors/daf/modal_job.py --stage env
modal run detectors/daf/modal_job.py --stage daf --weight anytext --size shrink512
modal run detectors/daf/modal_job.py --stage daf --weight anytext --size full
modal run detectors/daf/modal_job.py --stage daf --weight diffste --size shrink512
modal run detectors/daf/modal_job.py --stage daf --weight diffste --size full
modal run detectors/daf/modal_job.py --stage check_daf --weight anytext --size shrink512
modal run detectors/daf/modal_job.py --stage check_daf --weight anytext --size full
modal run detectors/daf/modal_job.py --stage check_daf --weight diffste --size shrink512
modal run detectors/daf/modal_job.py --stage check_daf --weight diffste --size full
```

Add `--pages page_0001,page_0003,page_0009` (one AI-edited, one handcrafted, one untouched page) to any `ocr`, `adcd`, `dtd` or `daf` call for the 3-page test. On Windows set `PYTHONUTF8=1` before `modal run`.

Mean time per page on a T4, from each run's `run_info.json` (about; model loading and ADCD-Net's OCR stage are extra):

| Detector | Downscaled | Full |
|---|---|---|
| ADCD-Net | 0.67 s (about 18 min for 1,650 pages) | 4.1 s (about 1 h 50 min) |
| DTD | 0.20 s (about 5 min) | 0.86 s (about 24 min) |
| DAF, AnyText weight | 0.31 s (about 9 min) | 1.50 s (about 41 min) |
| DAF, DiffSTE weight | 0.34 s (about 9 min) | 1.18 s (about 33 min) |

Cost on the shared team credit was small (ADCD-Net full size was estimated at about US$1.4 of T4 time from wall-clock time; the actual bill was not checked).

### 5. Copy the outputs, check them, score them

Each detector writes `out/<detector>/<size>/` with 1,650 `.npz` files, `run_info.json` and `run_log.csv`. Detector folder names: `adcd`, `dtd`, `daf_anytext`, `daf_diffste`. Full instructions: [`shared/README.md`](shared/README.md).

```bash
# copy both sizes and the sample list to a private folder (never download /pages)
modal volume get a4-data /out/<detector>/shrink512 <private>/outputs/<detector>
modal volume get a4-data /out/<detector>/full <private>/outputs/<detector>
modal volume get a4-data /meta/sample.csv <private>/sample.csv
# real-page check: must say PASS for both sizes
python3 shared/check_outputs.py --detector <detector> --sample <private>/sample.csv --out-root <private>/outputs
# score both sizes in one run (one run only keeps its own sizes in scores.csv)
python3 shared/score.py --detector <detector> --sample <private>/sample.csv --out-root <private>/outputs --results-dir results/<detector>
```

`score.py` writes `scores.csv` (every metric with its 95% CI and counts, with breakdowns), `differences.csv` (AI minus handcrafted, and downscaled minus full) and `score_info.json` (cut-offs, missing pages, a copy of `run_info.json`). In `score_info.json`, `pages_missing` must be 0 for both sizes.

### 6. Combined tables and charts

```bash
# cross-detector tables, then the charts (same commands as Level A)
python3 analysis/combined/make_tables.py
python3 analysis/figures/make_figures.py --out-dir results/combined/figures
```

### 7. People vs detectors (needs the private rating files)

```bash
# join the two raters' scores to anonymous page IDs; writes a private joined file (asks for its folders)
python3 analysis/human/join_ratings.py
# score people vs detectors; at the "Detectors" prompt type: adcd dtd daf_anytext daf_diffste
python3 analysis/human/people_vs_detectors.py
# draw the two charts
python3 analysis/human/make_figures.py
```

Before using rule B, `people_vs_detectors.py` rebuilds it on all 1,650 pages and stops unless every value equals `results/<detector>/scores.csv`. Details: [`analysis/human/README.md`](analysis/human/README.md).

### 8. Week 7 ablation study (needs the private Week 7 result files, no GPU)

```bash
# rescore the Week 7 per-page tables with rules A and B, AUC and edit found
python3 analysis/ablation/score_ablation.py <adcd>/per_page.csv <dtd>/per_page.csv <daf_anytext>/per_page.csv <daf_diffste>/per_page.csv
# paired contrasts (TextCtrl set only)
python3 analysis/ablation/make_contrast_summary.py <adcd>/per_page.csv <dtd>/per_page.csv <daf_anytext>/per_page.csv <daf_diffste>/per_page.csv
```

### 9. Extra checks (private outputs)

```bash
# is page size a shortcut for the label? (ADCD-Net full-size outputs; output committed as results/adcd/tile_check/tile_count_vs_label.txt)
python3 analysis/tile_check/tile_count_vs_label.py --sample <private>/sample.csv --folder <private>/outputs/adcd/full --scores results/adcd/scores.csv
# does the top score rise with tile count? (output committed as results/adcd/tile_check/tile_count_vs_top.txt)
python3 analysis/tile_check/tile_count_vs_top.py --folder <private>/outputs/adcd/full --folder <private>/outputs/adcd/shrink512 --sample-csv <private>/sample.csv
# spread of DTD's top scores
python3 -I analysis/dtd/top_score_spread.py --private-dir <private>
```

## Fixed settings

| Setting | Value |
|---|---|
| Random seed (sampling, document folds, bootstrap) | 42 |
| 95% CIs | 1,000 bootstrap resamples of documents (pages of one document stay together) |
| Flag threshold (rule A) | score ≥ 0.5, 8-connected regions |
| Rule B cut-off | chosen by 2-fold cross-fitting over documents, never on the pages it scores |
| Input sizes | Downscaled (longest side 512) and full (tiles: 512×512 for ADCD-Net and DTD, 1536×1024 for DAF, no overlap) |
| Capture condition | from each page's label: native, jpeg, resize_jpeg, scan_sim (simulated scan) |
| Test pages for every first run | `page_0001`, `page_0003`, `page_0009` |

## How to tell a Level B reproduction worked

Scoring all 1,650 pages should give 1,006 edited pages (1,093 edits), 644 untouched pages, 644 documents and 0 missing pages, and the AUC values in [Results at a glance](#results-at-a-glance) (all from `results/combined/tables/detector_table.csv`).

## Write-up figures and checks

The write-up points to this repo for its figures and for the page-size check. Both are here:

- **Figures 2 to 5** were drawn with matplotlib by [`analysis/figures/make_writeup_figures.py`](analysis/figures/make_writeup_figures.py) from `results/<detector>/scores.csv` and `results/human/auc.csv`. Its output is in `results/writeup/figures/`: the four PNGs (300 dpi) and `figure_data.csv`, which lists all 67 plotted values with their 95% CIs. Run with matplotlib 3.11.2 and the TeX Gyre Termes font (part of TeX Live), it redraws PNGs byte-identical to the committed ones; without that font the lettering falls back to matplotlib's default serif and the values are unchanged. `--pdf --out-dir <folder outside the repo>` also writes the PDFs used in the paper. Figure 1 (the pipeline) was drawn with TikZ inside the paper's LaTeX source.
- **The page-size check** (Section 2.3 of the write-up: AUC of tile count alone, edited vs untouched pages) comes from [`analysis/tile_check/tile_count_vs_label.py`](analysis/tile_check/tile_count_vs_label.py). Its printed output, run on the private ADCD-Net full-size outputs, is saved in `results/adcd/tile_check/tile_count_vs_label.txt`, together with `tile_count_vs_top.txt` (top score against tile count). The output's last line cross-checks the overall AUC against `results/adcd/scores.csv`.

## Known limits and what cannot be reproduced

- **Private inputs.** The pages, rating files and Week 7 result files cannot be shared, so Level B can only be run by the team and the mentor.
- **No heat maps are kept.** Each page keeps only its top score and the pixels at 0.5. This is all the scoring needs, but a different threshold or a pixel-level analysis would need the detectors rerun.
- **The Modal Volume will be deleted after the study** (our mentor's condition for using the programme's shared GPU workspace). The compact outputs are kept privately by the team.
- **The page-size check reads private outputs.** Its printed results are committed (`results/adcd/tile_check/`), but rerunning it needs the ADCD-Net outputs, which are not shared.
- **Two sets of charts.** The PNGs under `results/<detector>/figures/`, `results/combined/figures/` and `results/human/figures/` are the analysis versions, with older labels. The write-up's Figures 2 to 5 are the ones in `results/writeup/figures/`.
- **Our own setup.** Full-size tiling is our own design. DTD reads PNG pages, and JPEG pages with a coarse quantisation table, through a quality-100 JPEG re-save; the re-save counts of the full-size DTD run were overwritten and are not recoverable (`docs/decisions.md`, 20 Nov, T14). DAF gives boxes, which are coarser than heat maps.
- **Unverified setups.** DTD and DAF were not checked against their published scores; ADCD-Net was checked only in an earlier version of our pipeline (Week 5, `prior_work/abby/week5_adcd/`).
- **Ablation study.** Its headline contrasts hold for the PNG copies; on the JPEG copies DTD at full size also separates the AI edit (see `analysis/ablation/README.md`). Week 7 numbers that used DAF's SRNet weight cannot be reproduced from the repo.
- **Stage 2 uses two raters** (Abby, Nazhan); the plan said three, but the other two rating files came from an older version of the rating site with no page names. The two raters' files also come from different site versions (`docs/decisions.md`, T49).
- **`prior_work/` was not re-run.** It is Weeks 3 to 7 code as it was run, with paths replaced by `PATH_TO_*` placeholders and notebook outputs cleared. A few small helper scripts from that period were not kept, so some notebooks may not run as they stand.
- **Environment.** Detector images pin the authors' package versions where they install (see `detectors/*/requirements.txt`); where they did not, the closest working version is used. The versions really installed are written to each run's `run_info.json`.

## Credits

- Bank statements supplied by AI Lens Sdn. Bhd., Kuala Lumpur, under an NDA. Stage 1 filtering help from Chan Yong Jing.
- Detector code and weights belong to their authors: ADCD-Net, DTD (DocTamper) and DAF (OSTF), pinned above. AI editors: AnyText, UDiffText, TextCtrl (and DiffSTE, as a DAF weight and in the ablation study). Full references are in the write-up.
- Code in this repository was written with the help of AI coding assistants (Claude Code), directed and checked by the team. Every commit is authored by the team member who made it.
- Thanks to the MYResearchGuide committee for the programme and the GPU credit.
