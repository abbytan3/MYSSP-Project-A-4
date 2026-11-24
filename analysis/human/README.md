# T49: people vs detectors (stage 2)

Compares what two raters (Abby, Nazhan) thought of each of the 500 real stage 2 pages (160 AI, 160 handcrafted, 180 authentic) with what each detector did on the same page: ADCD-Net, DTD and DAF (AnyText and DiffSTE weights, shown separately), each downscaled (`shrink512`) and full size. Document level only: no pixel or location scores, no rater-agreement statistic. Method choices are in `docs/decisions.md` (lines marked T49). Only two raters are used: the other two stage 2 rating files came from an older version of the rating site with no page names, so they cannot be joined to pages.

## Run (both scripts ask for their paths; nothing about a laptop is in the code)

```
python3 analysis/human/join_ratings.py        # private: ratings + sample_map.csv -> joined file outside the repo
python3 analysis/human/people_vs_detectors.py # tables into results/human/
python3 analysis/human/make_figures.py        # charts (PNG) into results/human/figures/
python3 -m pytest analysis/human -q           # tests on small made-up arrays
```

Needs numpy, scipy (see `shared/requirements.txt`) and matplotlib for the charts.

1. `join_ratings.py` needs the private `sample_map.csv` and the ratings files (real file names). It prints counts only and writes `t49_joined_ratings.csv` (page_id, group, rater, score, no_traces, edit_found, false_alarm) to a private folder, and refuses a folder inside a git repo.
2. `people_vs_detectors.py` reads that file, `sample.csv` and the detector outputs. Before using rule B it rebuilds rule B on all 1,650 pages with `shared/score.py` and stops unless every point value equals `results/<detector>/scores.csv` and the cut-offs equal `score_info.json`. To add DTD or DAF, name it at the "Detectors" prompt; its outputs must be in `<out-root>/<detector>/` and its scores in `results/<detector>/`.

## Tables in `results/human/` (numbers only, no page IDs)

Every number has its 95% CI (1,000 bootstrap resamples of documents, seed 42) and its n. `small_n = True` means fewer than 30 pages, so the CI is shaky. In `detector_auc_by_people.csv` the "people missed" rows for AI edits rest on only 10 pages (both raters gave a 5): they are shaky, suggest nothing firm and should not be interpreted, even where an AUC looks high.

| File | What it holds |
|---|---|
| `rating_summary.csv` | per group and rater: n, mean score, share given a 5. For `mean_of_two`: mean of the two scores, and the share where **both** gave a 5 |
| `auc.csv` | AUC for edited vs authentic, AI vs authentic, handcrafted vs authentic: people (mean, each rater) and each detector size |
| `auc_differences.csv` | people (mean) AUC minus detector AUC, paired CIs; `clear` = CI excludes 0 |
| `detector_auc_by_people.csv` | detector AUC on edits people flagged / missed / in between, against the 180 authentic pages |
| `flag_rates.csv` | share of each group flagged by people or by rule B. On `authentic` this is the false-alarm rate |
| `edit_found_rates.csv` | descriptive: of the edited pages a rater flagged (score 2 or lower), how often their click hit the edit |
| `analysis_info.json` | seed, definitions, counts, the rule B reproduction check |

Direction: people score 5 = looks genuine, so people "edited-ness" is 6 minus the score; the detector top score is already high = edited. "People flagged" = mean score 2.5 or lower (one rater: 2 or lower). "People missed" = both raters gave a 5. A 3 means unsure and is not flagged.

## Notes

- The test `test_joined_file_has_only_the_allowed_columns_and_never_goes_inside_the_repo` checks that `join_ratings.py` refuses to write inside a git repo. Run the tests from a `git clone`: in a ZIP download there is no `.git` folder, so this one test fails and leaves a `should_not_exist/` folder (holding a one-row made-up test file) behind; delete it.
- The two charts were drawn during the analysis and their axis labels use older wording for the AUC = 0.5 line; the write-up's version is Figure 5 in `results/writeup/figures/`, redrawn from the same table (`results/human/auc.csv`) with the final wording.
- Headline values (people vs every detector setting) are collected in `results/README.md`.
