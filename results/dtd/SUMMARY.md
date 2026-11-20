# DTD analysis summary (T17)

DTD (CVPR 2023, released DocTamper weights) scored by `shared/score.py`. Numbers come from `results/dtd/scores.csv` and `results/dtd/differences.csv`, except the score-ceiling section (see its note). Ranges are 95% bootstrap ranges over documents. This file says what the numbers show, not why. All pages: 1,006 edited pages (1,093 edits: 541 AI, 552 handcrafted) and 644 untouched pages, at both sizes. Tables only; no figures were drawn for DTD.

## RQ1: detection

AUC, rule B recall and rule B false-alarm rate (rule B cut-offs by 2-fold cross-fitting). 644 untouched pages in every row.

| Size | Group | Edited pages | AUC | Rule B recall | Rule B false-alarm rate |
|---|---|---|---|---|---|
| shrink512 | all | 1,006 | 0.507 (0.482 to 0.532) | 0.992 (0.985 to 0.997) | 0.995 (0.989 to 1.000) |
| shrink512 | AI | 506 | 0.516 (0.486 to 0.544) | 0.996 (0.990 to 1.000) | 0.995 |
| shrink512 | handcrafted | 500 | 0.498 (0.470 to 0.525) | 0.988 (0.977 to 0.996) | 0.995 |
| full | all | 1,006 | 0.505 (0.478 to 0.532) | 0.998 (0.995 to 1.000) | 0.997 (0.992 to 1.000) |
| full | AI | 506 | 0.535 (0.505 to 0.564) | 0.996 (0.990 to 1.000) | 0.997 |
| full | handcrafted | 500 | 0.475 (0.444 to 0.506) | 1.000 (1.000 to 1.000) | 0.997 |

- **Rule A** (any pixel at or above 0.5), all pages: at shrink512 it flags every page (recall 1.000, false-alarm rate 1.000). At full, recall is 0.847 (0.821 to 0.872) and the false-alarm rate is 0.829 (0.800 to 0.857).
- **AI minus handcrafted:**

| Size | Metric | Difference | Range | Range excludes 0? |
|---|---|---|---|---|
| shrink512 | AUC | +0.018 | -0.009 to +0.045 | no |
| shrink512 | rule B recall | +0.008 | -0.000 to +0.018 | no |
| full | AUC | +0.060 | +0.032 to +0.085 | **yes** |
| full | rule B recall | -0.004 | -0.010 to +0.000 | no |

- **shrink512 minus full, all pages:** AUC +0.002 (-0.036 to +0.037, includes 0); rule B recall -0.006 (-0.014 to 0.000, includes 0); rule A recall +0.153 (+0.128 to +0.179, excludes 0).
- The false-alarm rate is on the same 644 untouched pages for both groups, so `differences.csv` has no AI minus handcrafted row for it.

## RQ2: location

An edit is found if a flagged region overlaps its box. False regions are counted on all pages.

| Size | Group | Edits found | Recall | Precision | False regions |
|---|---|---|---|---|---|
| shrink512 | all | 1,071 of 1,093 | 0.980 (0.967 to 0.990) | 0.105 (0.100 to 0.110) | 9,164 |
| shrink512 | AI | 531 of 541 | 0.982 (0.969 to 0.992) | 0.074 (0.069 to 0.078) | 6,677 |
| shrink512 | handcrafted | 540 of 552 | 0.978 (0.963 to 0.991) | 0.077 (0.072 to 0.082) | 6,481 |
| full | all | 400 of 1,093 | 0.366 (0.332 to 0.403) | 0.011 (0.010 to 0.013) | 34,754 |
| full | AI | 241 of 541 | 0.445 (0.403 to 0.490) | 0.010 (0.009 to 0.011) | 24,593 |
| full | handcrafted | 159 of 552 | 0.288 (0.248 to 0.330) | 0.006 (0.005 to 0.008) | 24,385 |

- **AI minus handcrafted, location recall:** shrink512 +0.003 (-0.009 to +0.017, includes 0); full +0.157 (+0.106 to +0.207, excludes 0).
- **shrink512 minus full, location recall, all pages:** +0.614 (+0.577 to +0.647, excludes 0).
- **False regions of the two groups** each include the false regions on the 644 untouched pages, so they do not add up to the "all" row. At shrink512, 6,677 + 6,481 = 13,158 against 9,164 (3,994 more). At full, 24,593 + 24,385 = 48,978 against 34,754 (14,224 more).

## RQ3: conditions

AUC with 95% range, shrink512 then full. n = edited pages; untouched pages are 644 for every editor/method group.

| Group | n edited (untouched) | shrink512 | full |
|---|---|---|---|
| AnyText | 177 | 0.527 (0.482 to 0.573) | 0.518 (0.472 to 0.562) |
| UDiffText | 204 | 0.495 (0.454 to 0.538) | **0.568 (0.527 to 0.605)** |
| TextCtrl | 125 | 0.533 (0.480 to 0.584) | 0.506 (0.452 to 0.558) |
| retypeset | 333 | 0.499 (0.466 to 0.532) | 0.480 (0.443 to 0.513) |
| copymove | 95 | 0.490 (0.429 to 0.548) | 0.471 (0.411 to 0.534) |
| pdf_edit | 72 | 0.506 (0.441 to 0.568) | 0.460 (0.392 to 0.525) |
| native | 181 (115) | 0.510 (0.448 to 0.577) | 0.541 (0.475 to 0.606) |
| jpeg | 409 (249) | 0.495 (0.448 to 0.539) | 0.509 (0.464 to 0.554) |
| resize_jpeg | 205 (150) | 0.507 (0.448 to 0.569) | 0.466 (0.400 to 0.527) |
| scan_sim | 211 (130) | 0.528 (0.467 to 0.591) | 0.547 (0.483 to 0.615) |

- No group has fewer than 30 pages (the smallest is pdf_edit, 72 edited pages).
- The only range that excludes 0.5 is UDiffText at full (bold). It is 1 of 20 ranges.
- The groups are shown side by side and are not compared with each other (`differences.csv` has no pairwise rows for editor, method or capture).

## Cross-detector context

All pages, from `results/<detector>/scores.csv`. Rule B false-alarm rate in the last column.

| Detector | Size | AUC | Rule B recall | Rule B false-alarm rate | Location recall | Location precision |
|---|---|---|---|---|---|---|
| DTD | shrink512 | 0.507 (0.482 to 0.532) | 0.992 | 0.995 | 0.980 | 0.105 |
| DTD | full | 0.505 (0.478 to 0.532) | 0.998 | 0.997 | 0.366 | 0.011 |
| ADCD-Net | shrink512 | 0.503 (0.486 to 0.522) | 0.998 | 0.995 | 0.101 | 0.013 |
| ADCD-Net | full | 0.533 (0.510 to 0.557) | 0.983 | 0.986 | 0.612 | 0.003 |
| DAF AnyText | shrink512 | 0.505 (0.487 to 0.522) | 0.997 | 0.995 | 0.156 | 0.010 |
| DAF AnyText | full | 0.598 (0.577 to 0.622) | 0.987 | 0.991 | 0.338 | 0.043 |
| DAF DiffSTE | shrink512 | 0.512 (0.492 to 0.530) | 0.999 | 1.000 | 0.284 | 0.007 |
| DAF DiffSTE | full | 0.595 (0.574 to 0.618) | 0.998 | 1.000 | 0.404 | 0.018 |

At shrink512 every detector's AUC is between 0.503 and 0.512. Only the two DAF weights at full are near 0.6.

## Score ceiling (DTD top scores)

**Note on source:** the two tables below (top score by side, and by stored file format) are in `results/dtd/top_score_spread.csv` (columns size, side, source, n, min, q1, median, q3, max). It is written by `analysis/dtd/top_score_spread.py`, which reads the private `.npz` outputs (each page's `top` value) and the group and format columns of `sample.csv`. The bullet counts (distinct values, pages at the maximum, pages at or above 0.90, between 0.5 and 0.90 and below 0.5, most common value) are printed by the same script and are not in the CSV. Numbers only, no page IDs.

Top score per page, 1,006 edited and 644 untouched pages:

| Size | Side | Min | Q1 | Median | Q3 | Max |
|---|---|---|---|---|---|---|
| shrink512 | edited | 0.9065 | 0.9107 | 0.9122 | 0.9138 | 0.9325 |
| shrink512 | untouched | 0.9074 | 0.9107 | 0.9120 | 0.9139 | 0.9284 |
| full | edited | 0.1272 | 0.6455 | 0.8733 | 0.9088 | 0.9287 |
| full | untouched | 0.1271 | 0.6172 | 0.8672 | 0.9104 | 0.9266 |

- **Not one repeated value.** At shrink512 there are 1,638 distinct top values over 1,650 pages, all between 0.9065 and 0.9325. One page sits at the maximum (0.9325). The most common value occurs 37 times. At full there are 1,253 distinct values and one page at the maximum (0.9287).
- **shrink512:** all 1,650 pages are at or above 0.90. The edited and untouched quartiles are almost the same.
- **full:** edited pages: 419 at or above 0.90, 433 between 0.5 and 0.90, 154 below 0.5. Untouched pages: 262, 272 and 110. The 264 pages below 0.5 are why rule A flags 0.847 (edited) and 0.829 (untouched) here, not every page.
- **By stored file format** (median, minimum and maximum):

| Size | Format | Side | Pages | Min | Median | Max |
|---|---|---|---|---|---|---|
| shrink512 | jpg | edited | 825 | 0.9065 | 0.9122 | 0.9325 |
| shrink512 | jpg | untouched | 529 | 0.9074 | 0.9121 | 0.9284 |
| shrink512 | png | edited | 181 | 0.9083 | 0.9119 | 0.9187 |
| shrink512 | png | untouched | 115 | 0.9080 | 0.9117 | 0.9192 |
| full | jpg | edited | 825 | 0.1272 | 0.8167 | 0.9287 |
| full | jpg | untouched | 529 | 0.1271 | 0.8216 | 0.9266 |
| full | png | edited | 181 | 0.9124 | 0.9183 | 0.9264 |
| full | png | untouched | 115 | 0.9117 | 0.9181 | 0.9259 |

At full, every PNG-source page has a top score of 0.9117 or more, edited and untouched alike. All the spread at full comes from JPEG-source pages.

## Read with care

- **Rule B flags nearly every page.** The false-alarm rate is 0.995 at shrink512 and 0.997 at full, almost the same as recall. Rule B recall alone says little.
- **AUC is at or near chance (0.5).** All pages: 0.507 (0.482 to 0.532) at shrink512 and 0.505 (0.478 to 0.532) at full, both ranges include 0.5.
- **Location recall comes with very large false-region counts** (9,164 at shrink512, 34,754 at full). Never quote recall without them. Precision is 0.105 and 0.011.
- **The AI vs handcrafted AUC gap exists at full only** (+0.060, range excludes 0; at shrink512 +0.018, range includes 0). It is small next to the ranges of single groups.
- **Rule A is not a result to compare sizes with.** It flags every page at shrink512 because every page's top score is above 0.5.
- **The scores are squeezed into a narrow band at shrink512** (0.9065 to 0.9325 for every page), and PNG-source pages are squeezed at full. A tampering signal would have to show up as a difference inside that band.
- **The link between the PNG pages at full size and the quality-100 re-save is a suggestion, untested.** PNG pages go through that re-save. The re-save counts for the full run could not be recovered (see `docs/decisions.md`, 20 Nov T14 lines), so it rests on the job code. No test was run that changes the re-save.
- **Run record.** `score_info.json` holds a copy of `run_info.json` taken after a zero-page test call reset its re-save counts to 0 (see `docs/decisions.md`).
- **No setup check against DTD's published scores** was done (docs/PLAN.md section 8).
