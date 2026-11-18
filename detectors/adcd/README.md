# ADCD-Net Modal job (T13)

Runs the released ADCD-Net weights on the sampled real pages at `shrink512` and `full`, and writes the compact output defined in `shared/output_format.py` (docs/PLAN.md section 3). Used exactly as released: no retraining.

## Files

| File | What it does |
|---|---|
| `modal_job.py` | The Modal job: OCR stage (CPU) and ADCD-Net stage (T4 GPU) |
| `tiling.py` | Cuts a page into 512x512 tiles and joins the tile maps back (rebuilt from Week 7 cell 25) |
| `test_tiling.py` | Local check of `tiling.py` (random arrays, no pages) |
| `requirements.txt` | The package versions used |

## Before the first run (Abby, once)

The weights and quantisation table go on the private Volume `a4-data` under `weights/adcd/`. Run these yourself (they use your Modal login; the agent never touches the token):

```
modal volume put a4-data "<Week 5 folder>/adcd-weights/ADCDNet.pth" /weights/adcd/ADCDNet.pth
modal volume put a4-data "<Week 5 folder>/adcd-weights/docres.pkl"  /weights/adcd/docres.pkl
modal volume put a4-data "<Week 5 folder>/04_adcd_run/adcd-run-outputs/qt_table.pk" /weights/adcd/qt_table.pk
```

The sample pages must already be on the Volume as `pages/page_0001.<ext>`, ... (the sampling and upload script in `shared/`).

## Order of runs

1. Test with 3 real pages (needs the "Help improve Claude" check only if an agent opens a page; the job itself prints no page content):
   ```
   modal run detectors/adcd/modal_job.py --stage ocr --limit 3
   modal run detectors/adcd/modal_job.py --stage adcd --size shrink512 --limit 3
   modal run detectors/adcd/modal_job.py --stage adcd --size full --limit 3
   ```
   Check `out/adcd/<size>/run_info.json` and `run_log.csv` on the Volume, then work out the cost from `median_seconds_per_page` and Modal's T4 price. Wait for a yes before the rest.
2. All pages: `--stage ocr` (no `--limit`), then `--stage adcd --size shrink512`, then `--stage adcd --size full`.

Every stage skips pages that already have a result, so a stopped run can simply be started again. Only one real run per detector at a time.

## Where things go on the Volume

- `adcd_work/ocr_masks/<page_id>.png`: page-sized text mask, one per page, used for both sizes. Never leaves the Volume.
- `adcd_work/ocr_logs/`: per-chunk OCR timing and PaddleOCR versions.
- `out/adcd/<size>/<page_id>.npz`, `run_info.json`, `run_log.csv`.
