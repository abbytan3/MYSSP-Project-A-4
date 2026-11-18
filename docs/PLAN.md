# PLAN.md: the confirmed experiment plan

Confirmed with the mentor (Jia Yu) on 18 Nov 2026. This file is written once and then frozen. Code is built from it. If anything here seems wrong, stop and ask the person; never change it on your own. Project-level status lives in the team's command centre sheet, not here.

## Title and research questions

**Can Off-the-Shelf Tampering Detectors Find AI Edits in Bank Statements?**

- **RQ1 (detection):** How accurately can pre-trained tampering detectors, used without fine-tuning, tell AI-edited bank statements apart from their untouched originals, and how often do they wrongly flag an untouched statement as tampered?
- **RQ2 (localisation):** How often do these detectors point to the field that was actually edited, and how many regions do they flag where nothing was changed, on both edited and untouched statements?
- **RQ3 (conditions):** Which factors change how well these detectors perform: the AI editor that made the edit, the page's capture condition, the input size (shrunk to 512 pixels or full size in tiles), or how convincing people judged the edit to be?

Handcrafted edits on the same kind of pages are the comparison point throughout: are AI edits harder or easier to catch than classic manual edits?

## The pipeline in one chain

```
Jia Yu's anonymised bank-statement dataset (stage 1 split, real pages only)
  → keep pages whose reviewer kept them in stage 1
  → sample: all 506 kept AI-edited pages + 500 kept handcrafted pages + their untouched originals (655)
  → rename to anonymous IDs (page_0001 …), upload to the private Modal Volume
  → run ADCD-Net, DTD, DAF (3 weights), released weights, at 512 and full size
  → heat maps (DAF: boxes) → flagged regions at 0.5
  → one shared scoring script: document level + location level, 95% ranges
  → breakdowns: AI vs handcrafted, editor/method, capture condition, input size, human grade
  → stage 2 human ratings on the same real pages: people vs detectors
  → Week 7 study rescored as ablation studies
```

## 1. Data

**Source.** Real Malaysian bank statements from the data partner, anonymised by Jia Yu (logos, names and details changed), already tampered by Jia Yu. The stage 1 split gives every page with its labels; the stage 1 decisions say which pages each reviewer kept.

- Stage 1 pages and labels (Abby's Mac): `stage1_split/reviewer{1-4}/set{1-4}/30_dataset/{images, masks, labels.jsonl}` and `stage1_split/reviewer{1-4}/assignment.csv`.
- Stage 1 decisions: `review_results_stage1/reviewer{1-4}/decisions_latest.json` (reviewer 3 has only `decisions.jsonl`: use the last decision per `id`). Reviewers: 1 Abby, 2 Alycia, 3 Nazhan, 4 a helper (credited in the acknowledgements). Each document went to exactly one reviewer.
- A page **passed stage 1** if its latest decision is `keep`.

**Scope: real statements only.** `set1` = real, handcrafted; `set2` = real, AI-edited. Synthetic sets (`set3`, `set4`) are not run through the detectors.

**The sample** (seed 42 for every random choice):

| Group | Rule | Count |
|---|---|---|
| AI-edited | Every `set2` tampered page that passed stage 1 | 506 (AnyText 177, UDiffText 204, TextCtrl 125) |
| Handcrafted | 500 `set1` tampered pages that passed stage 1. First include all 160 that appear in stage 2; fill the rest at random within each method so the totals match the passed pool's shares | 500 (retypeset 333, copy-move 95, PDF edit 72) |
| Untouched originals | The authentic page (`label = authentic`, file ending `_c0`) of every document behind a sampled tampered page, plus every real authentic page in stage 2. Each counted once | about 655 |

- The 1,383 handcrafted pages that passed have retypeset 923, copy-move 262, PDF edit 198. The 160 stage 2 handcrafted pages are 54, 53 and 53, so the random fill is uneven by design to hit 333 / 95 / 72.
- Pages with 2 edits are kept; every edited location is scored.
- Each page keeps its own **capture condition** from its labels (`native`, `jpeg`, `resize_jpeg`, `scan_sim`). No extra PNG/JPEG copies are made.
- **Ground truth:** `boxes` (`[x1, y1, x2, y2]` in page pixels) and `box_info` from `labels.jsonl`; masks are available but the boxes are the reference.
- **Anonymous IDs.** The sampling script writes `sample_map.csv` (anonymous ID → real file name, set, method, editor, capture, doc ID). It never enters the repo; it stays on Abby's Mac and on the Modal Volume. The repo only ever holds anonymous IDs. Real file names start with a bank code.

## 2. Storage and compute

- **Modal Volume `a4-data`** (private, team workspace): `pages/<page_id>.<ext>`, `meta/sample.csv` (anonymous ID, group, editor or method, capture, n_edits, boxes), `maps/<detector>/<size>/<page_id>.npy`, `results/`.
- Abby uploads the sample once. Teammates never download pages; their detector jobs read from the Volume.
- GPU: start on T4. Time 20 fake pages per detector and get a yes before the real run.
- Size of the run: about 1,661 pages × 2 sizes × 5 detector weights ≈ 16,600 detector passes.

## 3. Detectors

| Detector | Owner | Weights | Notes |
|---|---|---|---|
| ADCD-Net (ICCV 2025) | Abby | Released `ADCDNet.pth` + `docres.pkl` | Needs OCR text masks (PaddleOCR) as in Week 5; re-saves pages itself |
| DTD (CVPR 2023) | Alycia | Released DocTamper weights | Reads JPEG only: re-save PNG pages at quality 100 just before reading, and log it |
| DAF (from the OSTF paper) | Nazhan | Released SRNet, AnyText and DiffSTE weights (team Drive) | Outputs boxes with scores |

- Used exactly as released. No retraining or fine-tuning.
- **Two input sizes for every detector:** `shrink512` (longest side 512 px) and `full` (full size, cut into each detector's own tile size, no overlap, joined back).
- **Output format (shared, required):** one heat map per page and size, the same height and width as the original page, `float16` values 0 to 1, saved as `maps/<detector>/<size>/<page_id>.npy`. DAF: fill each box with its score (keep the maximum where boxes overlap), 0 elsewhere. Every job also writes `run_info.json` (weights, sizes, tile size, versions, seconds per page) and `run_log.csv` (page_id, size, seconds, max_value, status).

## 4. Scoring (one shared script for all detectors)

- **Flags:** pixels with score ≥ 0.5 (authors' default). Touching flagged pixels (8-connectivity) form one **region**.
- **Document level** (each page is "tampered" or "clean"):
  - Rule A: tampered if any pixel is flagged. Reported to show how the authors' default behaves (in Week 7 it flagged 84 to 100% of untouched pages); not the headline.
  - Rule B: tampered if the page's top score passes a cut-off. The cut-off is chosen by **2-fold cross-fitting**: split documents at random into two halves; choose the cut-off that maximises F1 on one half and apply it to the other; swap; report the pooled result. Test pages never choose their own cut-off. **Rule B is the headline document-level result.**
  - AUC from each page's top score (no cut-off needed).
  - Report precision, recall, F1, and the false-alarm rate on untouched pages.
- **Location level** (tampered pages, and untouched pages for false regions):
  - An edit is **found** if at least one region overlaps its box.
  - A **false region** is a region that overlaps no box, on any page.
  - Location precision = found edits ÷ (found edits + false regions); recall = found edits ÷ all edits; F1 from these.
- **Uncertainty:** 95% ranges from 1,000 bootstrap resamples of documents (pages from the same document stay together), seed 42. A difference counts as clear only if its range excludes 0. Every number is reported with its range and its count.
- **No pixel-level scores** are reported as findings.
- **Breakdowns:** AI vs handcrafted; AI editor (AnyText, UDiffText, TextCtrl); handcrafted method (retypeset, copy-move, PDF edit); capture condition; input size; stage 2 human grade.

## 5. Stage 2: people vs detectors

- Stage 2 used 1,000 pages (500 real, 500 synthetic: 160 handcrafted, 160 AI and 180 authentic on each side). Every rater saw the same pages, plus 30 pages shown twice (a consistency check), so each complete file has 1,030 ratings.
- **Raters used: Abby, Nazhan and Alycia** (`abby_`, `nazhan_`, `alycia_ratings_latest.json`, same layout). The helper's ratings come from an older version of the rating site with no page names, so they are left out unless a matching page list turns up. The analysis must run with whichever of these files exist.
- Fields used: `item_id` (`set/file name`, the join key with stage 1 labels), `label`, `score`, `no_traces`, `edit_found`, `false_alarm`. **Score: 5 = looks genuine, 4 = probably genuine, 3 = unsure, 2 = probably edited, 1 = clearly edited.** "No traces" is a 5.
- For a page shown twice, use the rater's later rating. A page's human score is the mean of the raters' scores.
- **Analysis (one only): people vs detectors.** On the 500 real stage 2 pages (all inside our sample), compare the human score and human "edit found" with each detector's top score and rule B decision on the same pages, for handcrafted vs AI and for authentic pages. No rater-agreement statistic.
- The ratings files hold real file names: they stay in Abby's private folder and on the Volume, never in the repo; only aggregated tables come out.

## 6. Ablation studies (Week 7)

Rescore the Week 7 controlled study from the safe `per_page.csv` files (no GPU) with the document-level rules and "location found": AI edits (TextCtrl, AnyText, DiffSTE) vs our recreations of generation, copy-move and splicing on the same pages, PNG vs twice-saved JPEG, 512 vs full size. False-region counts can't be recomputed (heat maps not kept) and are left out.

## 7. What goes in the repo

- Code, `run_info.json`, score tables with anonymous IDs only, and charts made from those tables (`results/<detector>/figures/`).
- Never: pages, crops, heat maps, `sample_map.csv`, real file names or doc IDs, bank codes, the Modal token, the ratings files' raw content if it holds file names.
- The write-up may show 1 to 2 anonymised synthetic sample pages (allowed by Jia Yu). They go in Overleaf only, never the repo.

## 8. Limitations to state

Real pages only; one document type; released weights only; one threshold (0.5); originals shared by several edits; fewer untouched than tampered pages; DTD's PNG pages re-saved at quality 100; DAF's boxes are coarser than heat maps; no setup checks for DTD and DAF against their published scores; stage 1 had one reviewer per document.
