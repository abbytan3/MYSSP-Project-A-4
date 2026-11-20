# Prior work: Alycia (Weeks 3 to 6 AnyText editing code)

Earlier code kept as a record, as it was run. It is **not** the final study: the final DTD job is `detectors/dtd/` (T14) and its top-score analysis is `analysis/dtd/` (T17). Nothing here is run by the repo or by `shared/`.

## Files

| File | What it did | What it produced | Which part of the submission uses it |
|---|---|---|---|
| `week3_6/make_tampered.py` | Made AnyText-edited copies of page images. It found the largest text box with EasyOCR (or read boxes from a JSON file), replaced the text with random new letters through the AnyText text-editing model, and kept the original and a ground-truth mask. | For each page: an untouched PNG, an edited PNG at the same size, a 0/255 box mask PNG, and a `manifest.json` with the box and the old and new text. | Background for how AnyText edits and their masks were made (AI-edit pages, docs/PLAN.md). |
| `week3_6/anytext_colab_test.ipynb` | A Google Colab starter notebook that checked the GPU, ran the AnyText smoke test, then edited one 512x512 crop of one page (a box mask plus new text, seed 1234, 20 steps). | Two test images in Colab (`crop.png`, `mask.png`) and the edited results. Nothing is saved in the notebook itself. | Same background: the AnyText test before the larger edit runs. |

## What was changed before it went into the repo

- Notebook: outputs cleared (it was saved with none). `jupyter nbconvert` is not installed on this laptop, so the cleared state was checked in the file itself (empty `outputs`, no execution counts).
- Paths replaced by placeholders: in `anytext_colab_test.ipynb`, the page path `/content/samples/stmt1.png` is now `PATH_TO_PAGE`.
- Left as they are: Colab's own `/content` folders, the public model name `damo/cv_anytext_text_generation_editing`, and the default folder names in `make_tampered.py` (`example_images`, `anytext_test`), which are not real paths.
- No pages, crops, masks, heat maps, `.npz`, CSVs with page names, weights, tokens or Modal workspace names are in this folder, and there are no document IDs or bank names.

Neither file was run again after cleaning (they need the private pages and a GPU).
