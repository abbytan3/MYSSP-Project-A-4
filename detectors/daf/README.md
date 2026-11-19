# DAF Modal job (T15)

Runs the released DAF weights (from the OSTF paper), AnyText and DiffSTE, on the sampled real pages at `shrink512` and `full`, and writes the compact output defined in `shared/output_format.py` (docs/PLAN.md section 3). Used exactly as released: no retraining. SRNet is not used. Follows the structure of `detectors/dtd/`.

DAF returns boxes with a tampered score. Each box is filled with its score into a page-sized map (the maximum where boxes overlap, 0 elsewhere), and only `top`, `shape` and `flag` are saved.

## Files

| File | What it does |
|---|---|
| `modal_job.py` | The Modal job: environment check (T4), DAF stage (T4), output check (CPU). Runs on Python 3.12 and starts `infer.py` |
| `infer.py` | The DAF code. Runs inside a separate Python 3.9 environment in the image (DAF needs 3.9) |
| `tiling.py` | Cuts a page into DAF-sized pieces and moves boxes back to page pixels |
| `test_tiling.py` | Local check of `tiling.py` (random arrays, no pages): `python detectors/daf/test_tiling.py` |
| `requirements.txt` | The package versions used and where they differ from the authors' list |

## Before the first run (Nazhan, once)

Two files go on the private Volume `a4-data` under `weights/daf/`. Run these yourself in your terminal (they use your Modal login; the agent never touches the token and uploads nothing without your OK):

```
modal volume put a4-data ~/Documents/daf-inputs/anytext.pth /weights/daf/anytext.pth
modal volume put a4-data ~/Documents/daf-inputs/diffste.pth /weights/daf/diffste.pth
```

The sample pages must already be on the Volume as `pages/page_0001.<ext>`, ... (the sampling and upload script in `shared/`). Only the Volume `a4-data` is used; the Modal workspace is shared with other teams.

## Order of runs

1. Environment check (T4, a few minutes, no pages): builds the image, loads both weight files and checks each reports **0 missing and 0 unexpected keys**, then runs one random-noise image:
   ```
   modal run detectors/daf/modal_job.py --stage env
   ```
2. Test with 3 real pages (`page_0001`, `page_0003`, `page_0009`), each weight at both sizes (four calls):
   ```
   modal run detectors/daf/modal_job.py --stage daf --weight anytext --size shrink512 --pages page_0001,page_0003,page_0009
   modal run detectors/daf/modal_job.py --stage daf --weight anytext --size full      --pages page_0001,page_0003,page_0009
   modal run detectors/daf/modal_job.py --stage daf --weight diffste --size shrink512 --pages page_0001,page_0003,page_0009
   modal run detectors/daf/modal_job.py --stage daf --weight diffste --size full      --pages page_0001,page_0003,page_0009
   ```
   Check `out/daf_<weight>/<size>/run_info.json` and `run_log.csv`, then work out the cost from `seconds_per_page` and Modal's current T4 price: 1,650 pages x 2 sizes x 2 weights = 6,600 passes. Wait for a yes before the rest.
3. All pages, `shrink512` for both weights first, then `full`: `--stage daf --weight <w> --size <size>` (no `--pages`). After each: `--stage check_daf --weight <w> --size <size>`. Only one real run at a time.
4. Then follow `shared/README.md` to copy the outputs, run `shared/check_outputs.py` (detector names `daf_anytext` and `daf_diffste`) and score both sizes in one `shared/score.py` run into `results/daf_anytext/` and `results/daf_diffste/`.

Every stage skips pages that already have a result, so a stopped run can simply be started again. A page that fails gets `error:<ExceptionType>` in `run_log.csv` and no `.npz`.

## How a page is read

- The page is read as raw BGR pixels (no EXIF turning). DAF's config resizes it to fit 1024 x 1536 keeping its shape (`max_per_img=1000`). The tampered score is column 1 of DAF's softmax.
- `shrink512`: longest side 512 (`INTER_AREA`); DAF then rescales that small copy itself (it is enlarged to fit 1024 x 1536). Boxes are scaled back to the page.
- `full`: pieces of 1536 (page's long side) x 1024 (short side) on a grid from (0,0), no overlap, white padding right and bottom. Boxes are moved by each piece's position and clipped to the page. A box that crosses a piece border is split in two.
- The weight-to-config pairing is `configs/fasterrcnn_x<weight>.py` in the OSTF repo (Faster R-CNN with Texture Jitter), pinned to the commit in `modal_job.py`.

## Where things go on the Volume

- `weights/daf/`: `anytext.pth`, `diffste.pth`.
- `out/daf_anytext/<size>/` and `out/daf_diffste/<size>/`: `<page_id>.npz`, `run_info.json`, `run_log.csv`.
