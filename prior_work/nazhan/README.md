# Prior work: Nazhan (Week 7 DAF run)

Earlier work kept as a record of how the DAF method was first run. It is **not** the final study: the final DAF run is `detectors/daf/` (task T15). Nothing here is run by the repo or by `shared/`.

## Files

| File | What it did | Which part of the submission uses it |
|---|---|---|
| `week7/DAF_Test.ipynb` | A Google Colab Pro notebook (T4 GPU, run top to bottom) that ran DAF (Faster R-CNN, Texture Jitter pre-trained, OSTF code) on every unique image of the Week 7 common page sets, with **three** weight files (`srnet`, `anytext`, `diffste`) at **three** input sizes (`shrink512`, `default`, `full`). It built a separate Python 3.9 environment, wrote one 8-bit result map per image, size and weight (each DAF box filled with 255 x its tampered score, maximum where boxes overlap), plus `boxes.csv`, `run_log.csv` and `run_info.json`, and zipped them for Abby's Week 7 scoring notebook. | The Week 7 ablation studies (docs/PLAN.md section 6, rescored by Abby from the per-page tables; the DAF AnyText and DAF DiffSTE results come from this run). It is also where the method for the final DAF job was worked out (config and weight pairing, the 1024x1536 test resize, the 1536x1024 pieces, softmax column 1 as the tampered score, the Python 3.9 environment); those choices are recorded in `docs/decisions.md` under T15. |

The file was renamed from `DAF Test.ipynb` to `DAF_Test.ipynb` (no space in the name). It is saved with `SMOKE = False`, the setting for the real run.

## Differences from the final study (so the two are not mixed up)

- **SRNet:** the SRNet weight (`srnet.pth`) was run in Week 7 but is **left out of the final study**, which uses only the AnyText and DiffSTE weights (docs/PLAN.md section 3; the ablation also excludes it, see `docs/decisions.md`, T20). The notebook still lists `srnet` because it is the Week 7 code as it was run.
- **Sizes:** Week 7 had a third size, `default` (the page as it is, DAF resizes it itself). The final study runs only `shrink512` and `full`.
- **Pages and output:** Week 7 used the Week 7 common page sets and 8-bit PNG maps; the final study uses the 1,650 sampled real pages and the compact `.npz` format of `shared/output_format.py` (float scores).

## What was changed before it went into the repo

- All outputs were cleared (cell outputs and run counters), so no page content, file name, count or hash from a real run is stored. Notebook metadata was cut down to the GPU type and the kernel name.
- Hard-coded paths and names replaced by placeholders:
  - the Google Drive folder that held the inputs and the results (in the settings cell and in the "how to run" text) is now `PATH_TO_DRIVE_FOLDER`;
  - the name of the Week 7 page-set zip file (5 places: the text, the zip search and its error message) is now `PAGES_ZIP_NAME`. Put the zip and the three `.pth` files in your Drive folder before running.
- Left as they are: Colab's own mount point `/content/drive`, Colab's local working folders under `/content`, and the public download links for the packages and the OSTF code.
- No pages, masks, weights, `.npz` files, tokens or Modal workspace names are in this folder, and the notebook has no document IDs or bank names.

The notebook has not been run again after cleaning (it needs the private inputs). It was only cleaned.
