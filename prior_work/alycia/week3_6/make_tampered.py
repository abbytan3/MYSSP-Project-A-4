"""Make tampered images with AnyText text-editing, keeping originals + GT masks.

Output layout (under --out):
  original/<name>.png   untouched copy (ground truth image)
  tampered/<name>.png   AnyText-edited image (same size as original)
  gt_mask/<name>.png    binary mask (255 = tampered region)
  manifest.json         per-image info (box, old text, new text)

Region selection:
  default      : EasyOCR finds text, the largest box is replaced
  --boxes FILE : JSON {"img.jpg": [x1, y1, x2, y2, "NEW TEXT"], ...} (no OCR needed)

Run on a machine with a GPU:
  pip install modelscope easyocr opencv-python  (+ AnyText deps, see its repo)
  python make_tampered.py --images example_images --out anytext_test
"""
import argparse, json, random, string
from pathlib import Path
import cv2
import numpy as np

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def ocr_largest_box(path, reader):
    best = None
    for pts, txt, conf in reader.readtext(str(path)):
        if conf < 0.5 or len(txt.strip()) < 2:
            continue
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        box = (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))
        area = (box[2] - box[0]) * (box[3] - box[1])
        if best is None or area > best[0]:
            best = (area, box, txt.strip())
    return None if best is None else (best[1], best[2])


def random_text(n, rng):
    n = max(2, min(n, 10))
    return "".join(rng.choice(string.ascii_uppercase) for _ in range(n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="example_images")
    ap.add_argument("--out", default="anytext_test")
    ap.add_argument("--boxes", help="JSON with manual boxes + new text")
    ap.add_argument("--caption", default="a photo of text")
    ap.add_argument("--pad", type=int, default=4, help="mask padding in px")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--steps", type=int, default=20)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out = Path(args.out)
    for d in ("original", "tampered", "gt_mask", "_work"):
        (out / d).mkdir(parents=True, exist_ok=True)

    manual = json.load(open(args.boxes)) if args.boxes else {}
    reader = None
    if not args.boxes:
        import easyocr
        reader = easyocr.Reader(["en"], gpu=True)

    from modelscope.pipelines import pipeline
    pipe = pipeline("my-anytext-task",
                    model="damo/cv_anytext_text_generation_editing",
                    model_revision="v1.1.1")

    manifest = []
    for p in sorted(Path(args.images).iterdir()):
        if p.suffix.lower() not in EXTS:
            continue
        img = cv2.imread(str(p))
        if img is None:
            print("skip unreadable", p.name)
            continue
        h, w = img.shape[:2]

        if p.name in manual:
            x1, y1, x2, y2, new = manual[p.name]
            old = None
        else:
            found = ocr_largest_box(p, reader) if reader else None
            if found is None:
                print("no text found, skip", p.name)
                continue
            (x1, y1, x2, y2), old = found
            new = random_text(len(old), rng)
            if new == old:
                new = new[::-1]
        x1, y1 = max(0, x1 - args.pad), max(0, y1 - args.pad)
        x2, y2 = min(w, x2 + args.pad), min(h, y2 + args.pad)

        mask = np.zeros((h, w), np.uint8)
        mask[y1:y2, x1:x2] = 255
        stem = p.stem
        src = out / "original" / f"{stem}.png"
        cv2.imwrite(str(src), img)
        mask_path = out / "gt_mask" / f"{stem}.png"
        cv2.imwrite(str(mask_path), mask)

        data = {
            "prompt": f'{args.caption}, "{new}"',
            "seed": args.seed,
            "draw_pos": str(mask_path),
            "ori_image": str(src),
        }
        params = {"show_debug": False, "image_count": 1, "ddim_steps": args.steps}
        results, rtn_code, rtn_warning, _ = pipe(data, mode="text-editing", **params)
        if rtn_code < 0:
            print("AnyText failed", p.name, rtn_warning)
            continue
        res = results[0]
        res = cv2.resize(res, (w, h), interpolation=cv2.INTER_LANCZOS4)
        # AnyText outputs RGB-like array; convert if colors look swapped in your env
        cv2.imwrite(str(out / "tampered" / f"{stem}.png"), res[:, :, ::-1])

        manifest.append({"name": stem, "box": [x1, y1, x2, y2],
                         "old_text": old, "new_text": new})
        print("done", p.name, "->", new)

    json.dump(manifest, open(out / "manifest.json", "w"), indent=2)
    print(f"{len(manifest)} pairs written to {out}")


if __name__ == "__main__":
    main()
