# DTD Modal job (T14)

Runs the released DTD weights (CVPR 2023, DocTamper) on the sampled real pages at `shrink512` and `full`, and writes the compact output defined in `shared/output_format.py` (docs/PLAN.md section 3). Used exactly as released: no retraining. Follows the structure of `detectors/adcd/`.

## Files

| File | What it does |
|---|---|
| `modal_job.py` | The Modal job: environment check (CPU), DTD stage (T4 GPU), output check (CPU) |
| `tiling.py` | Cuts a page into 512x512 tiles, joins the tile maps back, pads the JPEG DCT numbers |
| `test_tiling.py` | Local check of `tiling.py` (random arrays, no pages): `python -I detectors/dtd/test_tiling.py` |
| `requirements.txt` | The package versions used and where they differ from the authors' list |

## Before the first run (Alycia, once)

Three files go on the private Volume `a4-data` under `weights/dtd/` (from the authors' Google Drive link in their `models/Readme.md`). Run these yourself in PowerShell (they use your Modal login; the agent never touches the token). Do not upload `dtd_sroie.pth`.

```
modal volume put a4-data "$env:USERPROFILE\Downloads\dtd_doctamper.pth" /weights/dtd/dtd_doctamper.pth
modal volume put a4-data "$env:USERPROFILE\Downloads\vph_imagenet.pt"   /weights/dtd/vph_imagenet.pt
modal volume put a4-data "$env:USERPROFILE\Downloads\swin_imagenet.pt"  /weights/dtd/swin_imagenet.pt
```

`vph_imagenet.pt` and `swin_imagenet.pt` are whole saved ImageNet backbone objects that the authors' model code loads by name when it is built; the DocTamper weights then overwrite their values. The quantisation table file (`qt_table.pk`) is not needed: the job reads each page's table from its own JPEG data. The sample pages must already be on the Volume as `pages/page_0001.<ext>`, ... (the sampling and upload script in `shared/`).

On Windows set `PYTHONUTF8=1` before `modal run`, or Modal's progress marks crash the console (`'charmap' codec can't encode`).

## Order of runs

1. Environment check (CPU only, no pages, nearly free): builds the image, checks the three weight files, loads the model and runs one random tile:
   ```
   modal run detectors/dtd/modal_job.py --stage env
   ```
2. Test with 3 real pages (`page_0001`, `page_0003`, `page_0009`) at both sizes:
   ```
   modal run detectors/dtd/modal_job.py --stage dtd --size shrink512 --pages page_0001,page_0003,page_0009
   modal run detectors/dtd/modal_job.py --stage dtd --size full      --pages page_0001,page_0003,page_0009
   ```
   Check `out/dtd/<size>/run_info.json` and `run_log.csv` on the Volume, then work out the cost from `seconds_per_page` and Modal's T4 price. Wait for a yes before the rest.
3. All pages: `--stage dtd --size shrink512` (no `--pages`), then `--stage dtd --size full`. After each: `--stage check_dtd --size <size>`.
4. Then follow `shared/README.md` to copy the outputs, run `shared/check_outputs.py` and score both sizes in one `shared/score.py` run into `results/dtd/`.

Every stage skips pages that already have a result, so a stopped run can simply be started again. Only one real run per detector at a time.

## How a page is read

- The page is turned to greyscale (PIL), as in the authors' loader, and fed as 3 identical channels. DTD also takes the luma JPEG DCT numbers (absolute value, clipped to 0..20) and the 8x8 quantisation table.
- PNG pages are re-saved as JPEG at quality 100 just before reading (after padding, for `full`). JPEG pages keep their own DCT numbers and table. `shrink512` pages are always re-saved. The counts are in `run_info.json`.
- `shrink512`: longest side 512 (`INTER_AREA`), padded white to 512x512, map resized back to the page with bilinear interpolation.
- `full`: 512x512 tiles on a grid from (0,0), no overlap, white padding right and bottom, joined back, padding cropped.

## Where things go on the Volume

- `weights/dtd/`: the three weight files above.
- `out/dtd/<size>/<page_id>.npz`, `run_info.json`, `run_log.csv`.
