"""T09: shared sampling script (docs/PLAN.md section 1).

Reads only metadata (labels.jsonl, assignment.csv, stage 1 decisions, stage 2
rating files). Never opens a page image. Picks the pages every detector runs on:

    AI-edited    every set2 tampered page that passed stage 1  (506)
    handcrafted  500 set1 tampered pages that passed stage 1   (333 / 95 / 72)
    untouched    the authentic page of every document behind a sampled
                 tampered page, plus every real authentic page in stage 2

Writes two files into --out-dir, which must be OUTSIDE the repo:
    sample.csv      anonymous IDs only: this is what goes to the Modal Volume
                    as meta/sample.csv
    sample_map.csv  anonymous ID -> real file name, path, set, method, editor,
                    capture, doc ID. PRIVATE: never goes in the repo.

Run (all paths are arguments, none is stored in the repo):
    python3 shared/sample.py --stage1-split <dir> --review-results <dir> \
        --stage2-ratings <file> [<file> ...] --out-dir <dir outside repo> [--test]
"""

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Seed for every random choice in this script (PLAN.md section 1).
SEED = 42

# Target number of handcrafted pages per method (PLAN.md section 1 table).
HANDCRAFTED_TARGETS = {"retypeset": 333, "copymove": 95, "pdf_edit": 72}

# Counts the plan expects, used for the printed comparison.
EXPECTED_AI = 506
EXPECTED_HANDCRAFTED = 500
EXPECTED_ORIGINALS_ABOUT = 655

# Names used in the data for each handcrafted method and AI editor.
METHOD_NAMES = {"m3_retypeset": "retypeset", "m1_copymove": "copymove", "m7_pdf_edit": "pdf_edit"}
EDITOR_NAMES = {"AnyText": "AnyText", "UDiffText": "UDiffText", "TextCtrl": "TextCtrl"}

# Column order of the two output files.
SAMPLE_COLUMNS = ["page_id", "doc_id", "group", "editor_or_method", "capture", "n_edits", "boxes", "ext"]
MAP_COLUMNS = [
    "page_id", "doc_id", "group", "editor_or_method", "capture", "n_edits", "boxes", "ext",
    "set", "file_name", "path", "real_doc_id", "stage2",
]


def _read_jsonl(path):
    # Read a .jsonl file line by line as UTF-8, skipping blank lines.
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def load_records(split_dir, data_dir="30_dataset"):
    """Read labels.jsonl and assignment.csv for set1 and set2 of every reviewer.

    data_dir is "30_dataset" in the stage 1 split and "31_filtered" in the stage 2 split.
    If a page key appears under several reviewers, the first one read is kept.

    Returns {key: record} where key is "set/<file name>" (the same key the
    stage 1 decisions and the stage 2 ratings use).
    """
    split_dir = Path(split_dir)
    records = {}
    for reviewer_dir in sorted(split_dir.glob("reviewer*")):
        # The assignment file gives each page's document ID.
        doc_of = {}
        with open(reviewer_dir / "assignment.csv", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                doc_of[f"{row['set']}/{Path(row['file']).name}"] = row["doc_id"]
        # Only the real sets: set1 handcrafted, set2 AI-edited.
        for set_name in ("set1", "set2"):
            images = reviewer_dir / set_name / data_dir
            for label in _read_jsonl(images / "labels.jsonl"):
                file_name = Path(label["image"]).name
                key = f"{set_name}/{file_name}"
                records.setdefault(key, {
                    "key": key,
                    "set": set_name,
                    "file_name": file_name,
                    "path": str(images / "images" / file_name),
                    "label": label["label"],
                    "model": label.get("model"),
                    "method": label.get("method"),
                    "capture": label["capture"]["condition"],
                    "n_edits": int(label["n_edits"]),
                    "boxes": label["boxes"],
                    "doc_id": doc_of[key],
                })
    return records


def load_decisions(results_dir):
    """Latest stage 1 decision per page id, over all reviewers: {key: 'keep' or 'reject'}.

    Uses decisions_latest.json when it exists; otherwise (reviewer 3) takes the
    last decision per id from decisions.jsonl, later timestamp winning.
    """
    decisions = {}
    for reviewer_dir in sorted(Path(results_dir).glob("reviewer*")):
        latest = reviewer_dir / "decisions_latest.json"
        if latest.exists():
            with open(latest, encoding="utf-8") as handle:
                for key, value in json.load(handle).items():
                    decisions[key] = value["decision"]
        else:
            newest = {}
            for value in _read_jsonl(reviewer_dir / "decisions.jsonl"):
                # A later (or equal) timestamp replaces an earlier decision.
                if value["id"] not in newest or value["timestamp"] >= newest[value["id"]]["timestamp"]:
                    newest[value["id"]] = value
            decisions.update({k: v["decision"] for k, v in newest.items()})
    return decisions


def load_stage2_items(rating_files):
    """Set of page keys ("set/<file name>") that appear in any stage 2 ratings file."""
    items = set()
    for rating_file in rating_files:
        with open(rating_file, encoding="utf-8") as handle:
            for value in json.load(handle).values():
                items.add(value["item_id"])
    return items


def build_sample(records, decisions, stage2_items, seed=SEED):
    """Apply the PLAN.md rules and return (rows, report).

    rows: one dict per sampled page, already shuffled and numbered page_0001...
    report: counts and warnings to print.
    """
    rng = random.Random(seed)
    passed = lambda key: decisions.get(key) == "keep"
    report = {"warnings": []}

    # AI-edited: every set2 tampered page that passed stage 1 (no random choice).
    ai_keys = sorted(
        k for k, r in records.items() if r["set"] == "set2" and r["label"] == "tampered" and passed(k)
    )

    # Handcrafted pool: set1 tampered pages that passed stage 1, grouped by method.
    pool = defaultdict(list)
    for k, r in sorted(records.items()):
        if r["set"] == "set1" and r["label"] == "tampered" and passed(k):
            pool[METHOD_NAMES[r["method"]]].append(k)

    # Stage 2 handcrafted pages that are in the pool are all included first.
    stage2_hc = {k for k in stage2_items if k in records and records[k]["set"] == "set1"
                 and records[k]["label"] == "tampered" and passed(k)}
    hc_keys = set(stage2_hc)
    for method, target in HANDCRAFTED_TARGETS.items():
        have = sum(1 for k in stage2_hc if METHOD_NAMES[records[k]["method"]] == method)
        rest = [k for k in pool[method] if k not in stage2_hc]
        need = target - have
        if need < 0 or need > len(rest):
            report["warnings"].append(f"cannot reach {target} {method} pages (stage 2 has {have}, pool leaves {len(rest)})")
            need = max(0, min(need, len(rest)))
        # Random fill, seed 42, from the sorted pool so the result is repeatable.
        hc_keys.update(rng.sample(rest, need))
    hc_keys = sorted(hc_keys)

    # Documents behind the sampled tampered pages, plus real authentic pages in stage 2.
    tampered_keys = ai_keys + hc_keys
    docs = {records[k]["doc_id"] for k in tampered_keys}
    stage2_authentic = {k for k in stage2_items if k in records and records[k]["label"] == "authentic"}
    docs |= {records[k]["doc_id"] for k in stage2_authentic}
    report["stage2_items_not_in_labels"] = len([k for k in stage2_items if k not in records and k.startswith(("set1/", "set2/"))])

    # One authentic page per document. The set1 and set2 copies are the same image,
    # so the set1 copy is used (first by sorted key).
    authentic_of = {}
    for k, r in sorted(records.items()):
        if r["label"] == "authentic" and r["doc_id"] in docs and r["doc_id"] not in authentic_of:
            authentic_of[r["doc_id"]] = k
    missing = docs - set(authentic_of)
    if missing:
        report["warnings"].append(f"{len(missing)} documents have no authentic page")
    original_keys = sorted(authentic_of.values())

    # Build one row per page.
    def make_row(key, group):
        r = records[key]
        if group == "ai":
            detail = EDITOR_NAMES[r["model"].split()[0]]
        elif group == "handcrafted":
            detail = METHOD_NAMES[r["method"]]
        else:
            detail = ""
        return {
            "key": key, "group": group, "editor_or_method": detail, "capture": r["capture"],
            "n_edits": r["n_edits"], "boxes": json.dumps(r["boxes"], separators=(",", ":")),
            "ext": Path(r["file_name"]).suffix.lstrip("."), "real_doc_id": r["doc_id"],
            "set": r["set"], "file_name": r["file_name"], "path": r["path"],
            "stage2": int(key in stage2_items),
        }

    rows = ([make_row(k, "ai") for k in ai_keys]
            + [make_row(k, "handcrafted") for k in hc_keys]
            + [make_row(k, "authentic") for k in original_keys])

    # Shuffle (seed 42) so the numbering reveals neither group nor file order, then number.
    rng.shuffle(rows)
    anon_doc = {}
    for number, row in enumerate(rows, start=1):
        row["page_id"] = f"page_{number:04d}"
        # Anonymous document ID, numbered in order of first appearance.
        row["doc_id"] = anon_doc.setdefault(row["real_doc_id"], f"doc_{len(anon_doc) + 1:04d}")
    report["counts"] = Counter(r["group"] for r in rows)
    report["ai_editors"] = Counter(r["editor_or_method"] for r in rows if r["group"] == "ai")
    report["hc_methods"] = Counter(r["editor_or_method"] for r in rows if r["group"] == "handcrafted")
    report["stage2_pages_in_sample"] = Counter(r["group"] for r in rows if r["stage2"])
    report["documents"] = len(anon_doc)
    return rows, report


def write_outputs(rows, out_dir, name_suffix=""):
    """Write sample.csv (anonymous) and sample_map.csv (private) into out_dir."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for file_name, columns in ((f"sample{name_suffix}.csv", SAMPLE_COLUMNS), (f"sample_map{name_suffix}.csv", MAP_COLUMNS)):
        with open(out_dir / file_name, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(sorted(rows, key=lambda r: r["page_id"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage1-split", required=True, type=Path)
    parser.add_argument("--review-results", required=True, type=Path)
    parser.add_argument("--stage2-ratings", required=True, type=Path, nargs="+")
    parser.add_argument("--stage2-split", type=Path, default=None,
                        help="Week 6 stage 2 split: only used for stage 2 pages missing from the stage 1 split")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--test", action="store_true", help="also write a 3-page test sample (one per group)")
    args = parser.parse_args()

    # Refuse to write the private map anywhere inside the repo.
    repo_root = Path(__file__).resolve().parent.parent
    if repo_root in (args.out_dir.resolve(), *args.out_dir.resolve().parents):
        sys.exit("--out-dir must be outside the repo (sample_map.csv holds real file names)")

    rating_files = [p for p in args.stage2_ratings if p.exists()]
    print(f"stage 2 rating files found: {len(rating_files)} of {len(args.stage2_ratings)}")
    records = load_records(args.stage1_split)
    added = 0
    decisions = load_decisions(args.review_results)
    stage2_items = load_stage2_items(rating_files)
    # Stage 2 pages whose documents are not in the stage 1 split are read from the stage 2 split.
    if args.stage2_split:
        for key, record in load_records(args.stage2_split, data_dir="31_filtered").items():
            if key in stage2_items and key not in records:
                records[key] = record
                added += 1
    rows, report = build_sample(records, decisions, stage2_items)

    # Every sampled page file must exist on disk (checked by path only, never opened).
    absent = [r["page_id"] for r in rows if not Path(r["path"]).exists()]
    write_outputs(rows, args.out_dir)
    if args.test:
        # Test sample: the lowest page ID of each group, same IDs as the full sample.
        small = [next(r for r in sorted(rows, key=lambda x: x["page_id"]) if r["group"] == g)
                 for g in ("ai", "handcrafted", "authentic")]
        write_outputs(small, args.out_dir, name_suffix="_test")

    # Print the counts and compare with PLAN.md.
    c = report["counts"]
    print(f"seed {SEED}; documents: {report['documents']}; page files missing on disk: {len(absent)}")
    print(f"AI-edited:   {c['ai']:>4}   (plan {EXPECTED_AI})   editors {dict(report['ai_editors'])}")
    print(f"handcrafted: {c['handcrafted']:>4}   (plan {EXPECTED_HANDCRAFTED})   methods {dict(report['hc_methods'])}")
    print(f"originals:   {c['authentic']:>4}   (plan about {EXPECTED_ORIGINALS_ABOUT})")
    print(f"stage 2 pages inside the sample: {dict(report['stage2_pages_in_sample'])}")
    print(f"stage 2 pages added from the stage 2 split: {added}")
    print(f"stage 2 set1/set2 items still not found in any label file: {report['stage2_items_not_in_labels']}")
    for warning in report["warnings"]:
        print("WARNING:", warning)
    # The plan says "about 655" originals: counted as matching only within 1% of that.
    originals_ok = abs(c["authentic"] - EXPECTED_ORIGINALS_ABOUT) <= 0.01 * EXPECTED_ORIGINALS_ABOUT
    if not originals_ok:
        print(f"originals: {c['authentic']} vs plan about {EXPECTED_ORIGINALS_ABOUT}: differs, accepted (see decisions.md)")
    ok = (c["ai"] == EXPECTED_AI and c["handcrafted"] == EXPECTED_HANDCRAFTED and originals_ok
          and not report["warnings"] and not absent)
    print("MATCHES PLAN" if ok else "DOES NOT FULLY MATCH PLAN (see lines above)")
    # Exit code 1 only for a real failure (AI or handcrafted count, warning, missing file); the accepted originals gap is not one.
    ok = (c["ai"] == EXPECTED_AI and c["handcrafted"] == EXPECTED_HANDCRAFTED and not report["warnings"] and not absent)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
