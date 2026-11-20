"""Make a tiny FAKE input for notebook 3 in the exact standard layout, so the notebook can be tested without real data.

Usage:  python3 make_fake_test.py <output_folder>

What it writes (4 pages per set, all 5 versions x png/jpg, one TextCtrl page without a copy-move donor on purpose):
  <out>/PATH_TO_W7_COMMON_SETS_DATASET/w7_sets/manifest.csv, config.json, masks/box/<textctrl|shared49>/<page>.png, images/... (tiny)
  <out>/adcd-results/adcd/        run_info.json, run_log.csv, <shrink512|full>/<set>/<version>/<format>/<page>.png
  <out>/dtd-results/dtd/          same, sizes shrink512 and full
  <out>/daf-anytext-results/daf_anytext/   same, sizes shrink512, full and default
  <out>/notes-dataset/decisions.md  (a copy of the real notes file, to test that it is found)
The result maps are random 8-bit greyscale PNGs the SAME size as the page (CLAUDE.md "STANDARD result").
daf_srnet and daf_diffste are NOT made, to test that missing detectors are skipped with a printed note.
"""
import json
import os
import shutil
import sys

import numpy as np
import pandas as pd
from PIL import Image

# the output folder comes from the command line
OUT = sys.argv[1]
# one fixed seed so the fake data are always the same
rng = np.random.default_rng(42)

# the page sizes (width, height) and boxes (x1, y1, x2, y2) of 4 TextCtrl pages and 4 shared pages (4, not 3, so that every size group and
# every OCR group has at least 2 pages and the unpaired contrasts can run; SMOKE mode then uses the first 3)
TC_PAGES = {"page_0001": ((240, 120), (150, 40, 190, 60)), "page_0002": ((260, 130), (100, 70, 150, 90)), "page_0003": ((220, 110), (20, 20, 100, 50)),
            "page_0004": ((250, 120), (30, 60, 90, 100))}
SH_PAGES = {"img01": ((250, 125), (60, 50, 110, 70)), "img02": ((230, 115), (120, 30, 160, 50)), "img03": ((270, 140), (40, 90, 120, 120)),
            "img04": ((260, 130), (150, 80, 200, 110))}
# versions and formats
VERSIONS = ["ai_edit", "generation", "copy_move", "splicing", "untouched"]
FORMATS = ["png", "jpg"]
# size groups and OCR correctness (deliberately 2 + 1 so the unpaired contrasts can run)
SIZE_GROUPS = ["small", "small", "large", "large"]
OCR_OK = [True, True, False, False]

# folders of the fake Kaggle dataset
ROOT = os.path.join(OUT, "PATH_TO_W7_COMMON_SETS_DATASET", "w7_sets")


# make a folder if it does not exist and return the path
def mk(*parts):
    # join the parts
    p = os.path.join(*parts)
    # create the folder
    os.makedirs(p, exist_ok=True)
    # give it back
    return p


# ---------------- manifest, masks, tiny images ----------------
# manifest rows
rows = []
# the three sets: (set name, folder that holds non-AI versions, pages)
for set_name, src_set, pages in [("textctrl", "textctrl", TC_PAGES), ("anytext", "shared49", SH_PAGES), ("diffste", "shared49", SH_PAGES)]:
    # each page of the set
    for k, (page, ((w, h), (x1, y1, x2, y2))) in enumerate(pages.items()):
        # the box mask (white box on black), saved once per (src_set, page)
        mdir = mk(ROOT, "masks", "box", src_set)
        # mask array
        m = np.zeros((h, w), np.uint8)
        # the white box
        m[y1:y2, x1:x2] = 255
        # save it
        Image.fromarray(m).save(os.path.join(mdir, page + ".png"))
        # every version and format
        for version in VERSIONS:
            # TextCtrl page_0003 has no copy-move donor (a skipped page)
            if set_name == "textctrl" and page == "page_0003" and version == "copy_move":
                continue
            # both formats
            for fmt in FORMATS:
                # AI edits live in the set's own folder, the rest in the shared folder
                folder = set_name if version == "ai_edit" else src_set
                # relative path
                rel = "images/%s/%s/%s/%s.%s" % (folder, version, fmt, page, fmt)
                # a tiny grey image so the file exists
                idir = mk(ROOT, "images", folder, version, fmt)
                # save it
                Image.fromarray(np.full((h, w, 3), 200, np.uint8)).save(os.path.join(ROOT, rel))
                # one manifest row
                rows.append(dict(set=set_name, page=page, version=version, format=fmt, path=rel, mask_path="masks/box/%s/%s.png" % (src_set, page),
                                 editor=set_name, kind="word", new_text="x", x1=x1, y1=y1, x2=x2, y2=y2,
                                 box_share=round((x2 - x1) * (y2 - y1) / (w * h), 6), size_group=SIZE_GROUPS[k],
                                 ai_ocr_text="x" if version == "ai_edit" else "", ai_ocr_correct=OCR_OK[k] if version == "ai_edit" else "",
                                 ai_ned=0.0 if version == "ai_edit" else "", untouched_rebuilt=False))
# write the manifest
pd.DataFrame(rows).to_csv(os.path.join(ROOT, "manifest.csv"), index=False)
# write a config.json like notebook 1 does
with open(os.path.join(ROOT, "config.json"), "w") as f:
    json.dump({"SMOKE": True, "fake": True}, f)

# ---------------- fake detector results ----------------
# how strongly each fake detector lights up the box (mean 8-bit value inside the box) for edited versions
STRENGTH = {"adcd": 130, "dtd": 90, "daf_anytext": 200}
# detectors, their sizes
RUNS = {"adcd": ["shrink512", "full"], "dtd": ["shrink512", "full"], "daf_anytext": ["shrink512", "full", "default"]}
# each detector goes into its own fake dataset folder
for run, sizes in RUNS.items():
    # the dataset folder, e.g. adcd-results/adcd/
    base = mk(OUT, run.replace("_", "-") + "-results", run)
    # run_info.json as in the contract
    with open(os.path.join(base, "run_info.json"), "w") as f:
        json.dump({"detector": run.split("_")[0], "weights": run.split("_")[1] if "_" in run else "", "sizes": sizes, "fake": True}, f)
    # run_log.csv with one row per image and size (kept short: only the header and one row)
    pd.DataFrame([dict(image="fake", size=sizes[0], seconds=0.0, max_value=0, status="ok")]).to_csv(os.path.join(base, "run_log.csv"), index=False)
    # the unique images (anytext/diffste share the shared49 files)
    uniq = pd.DataFrame(rows).drop_duplicates("path")
    # every size and image
    for size in sizes:
        # every unique image
        for r in uniq.to_dict("records"):
            # path parts: images / set / version / format / file
            _, iset, version, fmt, fname = r["path"].split("/")
            # the page size
            pages = TC_PAGES if iset == "textctrl" else SH_PAGES
            # width, height and box of this page
            (w, h), (x1, y1, x2, y2) = pages[r["page"]]
            # background: random low values (a quiet detector)
            arr = rng.integers(0, 60, size=(h, w)).astype(np.uint8)
            # a few pages stay all zero (edge case: empty map)
            if rng.random() < 0.05:
                arr[:] = 0
            # edited versions: raise the box (random, so F1 varies); untouched stays quiet
            if version != "untouched":
                # box strength depends on the detector, a little on the size and format
                mean_box = STRENGTH[run] + (15 if size == "full" else 0) - (20 if fmt == "jpg" else 0)
                # random values around that strength
                inside = np.clip(rng.normal(mean_box, 50, size=(y2 - y1, x2 - x1)), 0, 255).astype(np.uint8)
                # DAF returns boxes: fill the box with one score instead of noise
                if run.startswith("daf"):
                    # one score for the whole box
                    inside[:] = np.uint8(np.clip(rng.normal(mean_box, 30), 0, 255))
                # write the box into the map
                arr[y1:y2, x1:x2] = inside
            # rare false alarm on an untouched page
            if version == "untouched" and rng.random() < 0.15:
                arr[0:3, 0:3] = 255
            # the output path follows the standard layout
            out_dir = mk(base, size, iset, version, fmt)
            # save the 8-bit greyscale PNG
            Image.fromarray(arr).save(os.path.join(out_dir, os.path.splitext(fname)[0] + ".png"))

# ---------------- decisions.md copy ----------------
# the real notes file (if it is next to this script's project)
real = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "notes", "decisions.md")
# a fake dataset that holds it
nd = mk(OUT, "notes-dataset")
# copy if it exists, otherwise write a one-line file
if os.path.exists(real):
    # copy
    shutil.copy(real, os.path.join(nd, "decisions.md"))
# no real file
else:
    # tiny file
    with open(os.path.join(nd, "decisions.md"), "w") as f:
        f.write("- 2026-01-01 00:00 — fake decision — test\n")
# done
print("fake input written to", OUT)
