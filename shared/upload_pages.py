"""T12: upload the sampled pages and meta/sample.csv to the private Modal Volume `a4-data`.

Layout written (docs/PLAN.md section 2):
    /pages/<page_id>.<ext>     one file per sampled page
    /meta/sample.csv           the anonymous sample (anonymous IDs only)
sample_map.csv (real file names) is NEVER uploaded; it is only read here to find
each page's local path. Pages are read in place and sent straight to the Volume:
no staging copy is made. Real names and paths are never printed.

Uses the Modal login that is already set up (never run `modal token`).

Run (all paths are arguments):
    python3 shared/upload_pages.py --sample-map <sample_map.csv> --sample <sample.csv> --test   # 3 pages
    python3 shared/upload_pages.py --sample-map <sample_map.csv> --sample <sample.csv>          # all pages
"""

import argparse
import csv
import sys
from pathlib import Path

import modal

# The one Volume this script may touch.
VOLUME_NAME = "a4-data"
# Pages per upload batch, and tries per batch.
BATCH_SIZE = 100
RETRIES = 3


def read_map(path):
    """Read sample_map.csv: returns a list of (page_id, ext, local_path)."""
    with open(path, encoding="utf-8", newline="") as handle:
        return [(r["page_id"], r["ext"], Path(r["path"])) for r in csv.DictReader(handle)]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sample-map", required=True, type=Path, help="private sample_map.csv (or sample_map_test.csv)")
    parser.add_argument("--sample", required=True, type=Path, help="anonymous sample.csv to upload as meta/sample.csv")
    parser.add_argument("--test", action="store_true", help="upload only the 3 pages in sample_map_test.csv next to --sample-map")
    args = parser.parse_args()

    # In test mode use the 3-page map that sample.py --test wrote next to the full map.
    map_path = args.sample_map.with_name("sample_map_test.csv") if args.test else args.sample_map
    pages = read_map(map_path)
    # Every local file must exist before anything is sent.
    missing = [pid for pid, _, local in pages if not local.is_file()]
    if missing:
        sys.exit(f"{len(missing)} page files missing locally, e.g. {missing[:3]}; nothing uploaded")

    volume = modal.Volume.from_name(VOLUME_NAME)
    # Send the pages in small batches: one huge batch of 1,650 files can drop the connection.
    # force=True lets a later upload overwrite files from an earlier (test or partial) one.
    for start in range(0, len(pages), BATCH_SIZE):
        chunk = pages[start:start + BATCH_SIZE]
        for attempt in range(1, RETRIES + 1):
            try:
                with volume.batch_upload(force=True) as batch:
                    for page_id, ext, local in chunk:
                        batch.put_file(str(local), f"/pages/{page_id}.{ext}")
                break
            except Exception as error:
                # Only the error type is printed: messages can contain local file paths.
                print(f"batch starting at page {start + 1}: attempt {attempt} failed ({type(error).__name__})")
                if attempt == RETRIES:
                    sys.exit("giving up; rerun the same command to continue (uploads are repeatable)")
        print(f"sent {min(start + BATCH_SIZE, len(pages))} of {len(pages)} pages", flush=True)
    # The anonymous sample file goes last.
    with volume.batch_upload(force=True) as batch:
        batch.put_file(str(args.sample), "/meta/sample.csv")
    print(f"uploaded {len(pages)} pages and meta/sample.csv")

    # Check the Volume: same file names and same byte sizes as the local files.
    remote = {Path(e.path).name: e.size for e in volume.listdir("/pages")}
    expected = {f"{pid}.{ext}": local.stat().st_size for pid, ext, local in pages}
    absent = sorted(set(expected) - set(remote))
    wrong_size = sorted(n for n in expected if n in remote and remote[n] != expected[n])
    print(f"Volume /pages holds {len(remote)} files; expected from this upload: {len(expected)}; "
          f"absent: {len(absent)}; size mismatches: {len(wrong_size)}")
    if absent or wrong_size:
        print("problem pages:", (absent + wrong_size)[:10])
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
