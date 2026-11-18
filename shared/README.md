# From a finished detector run to scored results

For the owner of a detector (ADCD-Net, DTD, DAF). Your run has written `out/<detector>/shrink512/` and `out/<detector>/full/` on the Modal Volume `a4-data`. Each folder holds 1,650 `.npz` files, `run_info.json` and `run_log.csv`. `<detector>` is the folder name under `out/`; if your detector has several weights, each has its own folder name and you repeat the steps for each. Do these steps in order. Your coding agent can run every command. The Volume is deleted at the end, so copy your outputs first.

`<private>` below means a folder **outside this repo**, for example `~/a4_private` (Windows: `C:\Users\you\a4_private`). Never inside the repo.

## 1. Copy the compact outputs and the anonymous sample list

```
modal volume get a4-data /out/<detector>/shrink512 <private>/outputs/<detector>
modal volume get a4-data /out/<detector>/full      <private>/outputs/<detector>
modal volume get a4-data /meta/sample.csv          <private>/sample.csv
```

- Copy only these. Never download `/pages`, never copy a page anywhere. Never run `modal token`.
- Each size should now have 1,652 files: 1,650 `.npz` plus `run_info.json` and `run_log.csv`.

## 2. Real-page check (must say PASS for both sizes)

```
python3 shared/check_outputs.py --detector <detector> --sample <private>/sample.csv --out-root <private>/outputs
```

It checks the file layout, that `run_info.json` has its 7 required fields (`detector`, `weights`, `size`, `tile_size`, `versions`, `seconds_per_page`, `seed`), that every page is there and `ok` in `run_log.csv`, and that a page counts as edited by rule A exactly when its top score is at least 0.5. If it says FAIL, stop and tell Abby. Do not edit the outputs or `score.py`.

## 3. Score both sizes in one run

```
pip install -r shared/requirements.txt
python3 shared/score.py --detector <detector> --sample <private>/sample.csv --out-root <private>/outputs --results-dir results/<detector>
```

Score **both sizes in the same run**. `scores.csv` only keeps the sizes of one run, so scoring one size later overwrites the other. It takes about a minute and prints the headline numbers with their 95% ranges.

## 4. Check what landed

`results/<detector>/` now holds `scores.csv` (every metric, with ranges and counts, plus breakdowns), `differences.csv` (AI minus handcrafted, and `shrink512` minus `full`) and `score_info.json` (seed, cut-offs, a copy of `run_info.json`, missing pages). Open `score_info.json`: `pages_missing` must be 0 for both sizes. Pages scored should be 1,650.

## 5. Commit

- **Commit:** the three files in `results/<detector>/`, plus a line in `docs/decisions.md` for any choice or problem. Start the message with your task ID, stage files by name, then `git fetch`, `git rebase origin/main`, `git push origin HEAD:main`.
- **Never commit:** pages, `.npz` files, heat maps, `sample_map.csv`, real file names or document IDs, anything from `<private>`, the Modal token. The pre-commit check blocks these. Never use `--no-verify`.
- **Charts (optional):** after results exist, `python3 analysis/figures/make_figures.py` draws the cross-detector charts into `analysis/figures/out/` (not committed).

When you finish, update your task on the command centre sheet.
