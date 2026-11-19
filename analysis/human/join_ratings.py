"""T49 step 2: join the stage 2 ratings to anonymous page IDs.

The ratings files hold real file names (item_id = "set/file name"). The only
link from a real name to an anonymous page ID (page_0001 ...) is the private
sample_map.csv. This script does that join and:

  * prints counts and anonymous page IDs only (never an item_id, file name or doc ID);
  * with a private output folder, writes one joined file there (anonymous page IDs and
    rating fields only, no item_id), for the analysis script to read.

Folder paths are asked for when the script runs. They are never written in code.
Leave the output folder blank for a dry run that only prints counts.
"""

import csv
import json
from collections import Counter
from pathlib import Path

# The two raters used (PLAN CHANGE: two raters, not three) and their file names.
RATERS = {"abby": "abby_ratings_latest.json", "nazhan": "nazhan_ratings_latest.json"}
# Sets 1 and 2 are real pages; sets 3 and 4 are synthetic and are left out.
REAL_SETS = ("set1", "set2")


def latest_per_item(records):
    """One rating per item_id. For a page shown twice, keep the later one.

    "Later" = the higher position in the rating session (position counts the order
    pages were shown). Returns (dict item_id -> record, number of repeated item_ids).
    """
    best, seen = {}, Counter()
    for rec in records:
        item = rec["item_id"]
        seen[item] += 1
        # Keep the record shown last (larger position).
        if item not in best or rec["position"] > best[item]["position"]:
            best[item] = rec
    repeats = sum(1 for n in seen.values() if n > 1)
    return best, repeats


def build_key_to_page(sample_map_rows):
    """Map "set/file name" (the ratings key) to the sample_map row."""
    return {f"{r['set']}/{r['file_name']}": r for r in sample_map_rows}


def join_one_rater(latest, key_to_page):
    """Join one rater's latest ratings to sample_map rows.

    Returns (joined rows as dicts with page_id, unmatched real count).
    """
    joined, unmatched_real = [], 0
    for item, rec in latest.items():
        if item.split("/")[0] not in REAL_SETS:
            continue
        row = key_to_page.get(item)
        if row is None:
            unmatched_real += 1
            continue
        joined.append({
            "page_id": row["page_id"],
            "group": row["group"],
            "stage2": row["stage2"],
            # label is used only for the printed agreement check, never written to the file
            "label": rec["label"],
            "score": rec["score"],
            "no_traces": rec["no_traces"],
            "edit_found": rec["edit_found"],
            "false_alarm": rec["false_alarm"],
        })
    return joined, unmatched_real


# Columns of the private joined file. No item_id, file name, clicks or timestamps.
JOINED_FIELDS = ["page_id", "group", "rater", "score", "no_traces", "edit_found", "false_alarm"]


def write_joined(per_rater, out_dir):
    """Write the joined ratings to a private folder. Refuses a folder inside a git repo."""
    out_dir = Path(out_dir)
    # Safety: the file must never end up in the repo.
    for parent in [out_dir.resolve(), *out_dir.resolve().parents]:
        if (parent / ".git").exists():
            raise SystemExit("Refusing to write inside a git repo. Pick a private folder outside it.")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "t49_joined_ratings.csv"
    n_rows = 0
    with open(out_file, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=JOINED_FIELDS)
        writer.writeheader()
        for who, rows in per_rater.items():
            for j in sorted(rows, key=lambda r: r["page_id"]):
                writer.writerow({"rater": who, **{k: j[k] for k in JOINED_FIELDS if k != "rater"}})
                n_rows += 1
    return out_file, n_rows


def ask(prompt):
    """Ask for a path at run time (paths are never stored in code)."""
    return input(prompt).strip().strip("'\"")


def main():
    ratings_dir = Path(ask("Folder with the ratings files: "))
    map_path = Path(ask("Path to sample_map.csv (private): "))
    sample_path = Path(ask("Path to the anonymous sample.csv: "))
    out_text = ask("Private output folder outside the repo (blank = dry run, counts only): ")

    # Read the private map and the anonymous sample list.
    with open(map_path, encoding="utf-8", newline="") as fh:
        map_rows = list(csv.DictReader(fh))
    with open(sample_path, encoding="utf-8", newline="") as fh:
        sample_ids = {r["page_id"] for r in csv.DictReader(fh)}
    key_to_page = build_key_to_page(map_rows)
    print(f"sample_map rows: {len(map_rows)}; anonymous sample rows: {len(sample_ids)}")
    print(f"sample_map page_ids all in sample.csv: {all(r['page_id'] in sample_ids for r in map_rows)}")
    print(f"stage2 pages in sample_map: {sum(r['stage2'] == '1' for r in map_rows)}")

    per_rater = {}
    for who, fname in RATERS.items():
        # Each file is {record id: rating record}, 1,030 records per rater.
        data = json.loads((ratings_dir / fname).read_text(encoding="utf-8"))
        records = list(data.values())
        latest, repeats = latest_per_item(records)
        real_items = [i for i in latest if i.split("/")[0] in REAL_SETS]
        joined, unmatched = join_one_rater(latest, key_to_page)
        per_rater[who] = joined
        print(f"\n== {who}")
        print(f"  records {len(records)}; distinct item_ids {len(latest)}; item_ids shown twice {repeats}")
        print(f"  real item_ids (set1/set2) {len(real_items)}; matched in sample_map {len(joined)}; unmatched {unmatched}")
        ids = {j["page_id"] for j in joined}
        print(f"  distinct page_ids {len(ids)}; all in sample.csv {ids <= sample_ids}")
        print(f"  on stage 2 list in sample_map: {sum(j['stage2'] == '1' for j in joined)} of {len(joined)}")
        print(f"  by group: {dict(Counter(j['group'] for j in joined))}")
        # Does the rating-file label agree with the sample group?
        agree = Counter((j['group'] == 'authentic') == (j['label'] == 'authentic') for j in joined)
        print(f"  rating label agrees with sample group: {dict(agree)}")

    # Both raters should cover the same pages.
    a, b = ({j["page_id"] for j in per_rater[w]} for w in RATERS)
    print(f"\npages both raters rated {len(a & b)}; only first rater {len(a - b)}; only second rater {len(b - a)}")
    if a - b or b - a:
        print("  example page_ids not shared:", sorted((a ^ b))[:5])

    if out_text:
        out_file, n_rows = write_joined(per_rater, Path(out_text))
        print(f"\nwrote {n_rows} rows to the private folder (file name {out_file.name}).")
    else:
        print("\nDry run: nothing written.")


if __name__ == "__main__":
    main()
