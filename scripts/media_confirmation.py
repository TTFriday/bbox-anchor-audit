"""Read-only media checks for archived bounding-box conventions.

The checks are diagnostics.  They do not rewrite labels and they do not claim
that a colour heuristic establishes a file format.  Use the metadata
fingerprint, source documentation and pixel/mask evidence together.

Examples::

    python media_confirmation.py greenness ARCHIVE --images 156
    python media_confirmation.py semantic ARCHIVE --images 156
    python media_confirmation.py instances ARCHIVE --field ground_truth --images 156
    python media_confirmation.py nesting ARCHIVE --child leaf --parent "whole plant"
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import random
import statistics
import zlib
from pathlib import Path

import numpy as np
from PIL import Image


def read_samples(directory):
    with (directory / "samples.json").open(encoding="utf-8") as stream:
        return json.load(stream)["samples"]


def rectangles(box, mode, width, height):
    x, y, w, h = box
    if mode == "centre":
        return ((x - w / 2) * width, (y - h / 2) * height,
                (x + w / 2) * width, (y + h / 2) * height)
    return (x * width, y * height, (x + w) * width, (y + h) * height)


def clipped(rect, width, height, minimum=1):
    x0, y0, x1, y1 = rect
    x0, y0 = max(0, int(round(x0))), max(0, int(round(y0)))
    x1, y1 = min(width, int(round(x1))), min(height, int(round(y1)))
    if x1 - x0 < minimum or y1 - y0 < minimum:
        return None
    return x0, y0, x1, y1


def green_fraction(array):
    a = array.astype(np.int16)
    return ((a[..., 1] > a[..., 0] + 5) & (a[..., 1] > a[..., 2] + 5)).astype(float)


def greenness(directory, images, field, seed):
    records = []
    for sample in read_samples(directory):
        label = sample.get(field) or {}
        boxes = [d["bounding_box"] for d in label.get("detections", [])
                 if isinstance(d.get("bounding_box"), list) and len(d["bounding_box"]) == 4]
        if boxes:
            records.append((sample["filepath"], boxes))
    random.Random(seed).shuffle(records)
    result = {mode: [] for mode in ("centre", "corner")}
    control = {mode: [] for mode in ("centre", "corner")}
    missing = 0
    for filepath, boxes in records[:images]:
        path = directory / filepath
        if not path.is_file():
            missing += 1
            continue
        with Image.open(path) as image:
            array = np.asarray(image.convert("RGB"))
        height, width = array.shape[:2]
        green = green_fraction(array)
        rng = np.random.RandomState(seed)
        for box in boxes:
            for mode in result:
                cell = clipped(rectangles(box, mode, width, height), width, height, 10)
                if cell is None:
                    continue
                x0, y0, x1, y1 = cell
                result[mode].append(float(green[y0:y1, x0:x1].mean()))
                for _ in range(3):
                    rx = int(rng.randint(0, max(1, width - (x1 - x0) + 1)))
                    ry = int(rng.randint(0, max(1, height - (y1 - y0) + 1)))
                    control[mode].append(float(green[ry:ry + y1 - y0, rx:rx + x1 - x0].mean()))
    return {"images_available": len(records[:images]) - missing, "missing_images": missing,
            "centre_gain": float(np.mean(result["centre"]) - np.mean(control["centre"])),
            "corner_gain": float(np.mean(result["corner"]) - np.mean(control["corner"])),
            "centre_boxes": len(result["centre"]), "corner_boxes": len(result["corner"]),
            "note": "Colour gain is supporting evidence only."}


def semantic(directory, images, box_field, mask_field, seed):
    records = []
    for sample in read_samples(directory):
        label, mask_label = sample.get(box_field), sample.get(mask_field)
        if not isinstance(label, dict) or not isinstance(mask_label, dict):
            continue
        mask_path = mask_label.get("mask_path")
        boxes = [d["bounding_box"] for d in label.get("detections", [])
                 if isinstance(d.get("bounding_box"), list) and len(d["bounding_box"]) == 4]
        if mask_path and boxes and (directory / mask_path).is_file():
            records.append((mask_path, boxes))
    random.Random(seed).shuffle(records)
    inside = {mode: [] for mode in ("centre", "corner")}
    ring = {mode: [] for mode in ("centre", "corner")}
    for mask_path, boxes in records[:images]:
        with Image.open(directory / mask_path) as image:
            mask = np.asarray(image)
        height, width = mask.shape[:2]
        values, counts = np.unique(mask, return_counts=True)
        foreground = mask != int(values[np.argmax(counts)])
        band = max(3, int(.02 * min(width, height)))
        for box in boxes:
            for mode in inside:
                cell = clipped(rectangles(box, mode, width, height), width, height, 8)
                if cell is None:
                    continue
                x0, y0, x1, y1 = cell
                inside[mode].append(float(foreground[y0:y1, x0:x1].mean()))
                ox0, oy0 = max(0, x0 - band), max(0, y0 - band)
                ox1, oy1 = min(width, x1 + band), min(height, y1 + band)
                annulus = foreground[oy0:oy1, ox0:ox1].copy()
                annulus[y0 - oy0:y1 - oy0, x0 - ox0:x1 - ox0] = False
                ring[mode].append(float(annulus.mean()))
    return {mode: {"inside_foreground": float(np.mean(inside[mode])),
                   "annulus_foreground": float(np.mean(ring[mode])),
                   "tightness": float(np.mean(inside[mode]) - np.mean(ring[mode])),
                   "boxes": len(inside[mode])} for mode in inside}


def decode_mask(detection):
    try:
        payload = detection["mask"]["$binary"]["base64"]
        raw = base64.b64decode(payload)
        return np.lib.format.read_array(io.BytesIO(zlib.decompress(raw)), allow_pickle=True)
    except (KeyError, ValueError, TypeError, zlib.error):
        return None


def instances(directory, images, field, seed):
    records = []
    for sample in read_samples(directory):
        label = sample.get(field) or {}
        detections = [d for d in label.get("detections", [])
                      if isinstance(d.get("bounding_box"), list) and len(d["bounding_box"]) == 4
                      and isinstance(d.get("mask"), dict)]
        if detections:
            records.append((sample["filepath"], detections))
    random.Random(seed).shuffle(records)
    scores = {mode: [] for mode in ("centre", "corner")}
    matched = total = tight = 0
    for filepath, detections in records[:images]:
        path = directory / filepath
        if not path.is_file():
            continue
        with Image.open(path) as image:
            array = np.asarray(image.convert("RGB"))
        height, width = array.shape[:2]
        green = green_fraction(array)
        for detection in detections:
            mask = decode_mask(detection)
            if mask is None:
                continue
            box = detection["bounding_box"]
            total += 1
            matched += int(mask.shape == (int(round(box[3] * height)), int(round(box[2] * width))))
            if mask.ndim == 2 and min(mask.shape) > 6:
                tight += float(np.mean([mask[0].any(), mask[-1].any(), mask[:, 0].any(), mask[:, -1].any()]))
            for mode in scores:
                cell = clipped(rectangles(box, mode, width, height), width, height, 4)
                if cell is None:
                    continue
                x0, y0, x1, y1 = cell
                resized = np.asarray(Image.fromarray((mask * 255).astype(np.uint8)).resize((x1 - x0, y1 - y0))) > 127
                if resized.any():
                    scores[mode].append(float(green[y0:y1, x0:x1][resized].mean()))
    return {"matched": matched, "total": total, "shape_match_rate": matched / max(1, total),
            "four_side_fit": tight / max(1, total),
            "centre_foreground": float(np.mean(scores["centre"])),
            "corner_foreground": float(np.mean(scores["corner"]))}


def nesting(directory, maximum, fields, child_labels, parent_labels):
    coverage = {mode: [] for mode in ("centre", "corner")}
    images = 0
    for sample in read_samples(directory):
        detections = []
        for field in fields:
            label = sample.get(field) or {}
            detections.extend(d for d in label.get("detections", [])
                              if isinstance(d.get("bounding_box"), list) and len(d["bounding_box"]) == 4)
        metadata = sample.get("metadata") or {}
        width, height = metadata.get("width") or 1, metadata.get("height") or 1
        parents = [d["bounding_box"] for d in detections if d.get("label") in parent_labels]
        children = [d["bounding_box"] for d in detections if d.get("label") in child_labels]
        if not parents or not children:
            continue
        images += 1
        for mode in coverage:
            p = [rectangles(box, mode, width, height) for box in parents]
            q = [rectangles(box, mode, width, height) for box in children]
            for child in q:
                best = 0.0
                for parent in p:
                    iw = max(0.0, min(child[2], parent[2]) - max(child[0], parent[0]))
                    ih = max(0.0, min(child[3], parent[3]) - max(child[1], parent[1]))
                    area = max(1e-9, (child[2] - child[0]) * (child[3] - child[1]))
                    best = max(best, iw * ih / area)
                coverage[mode].append(best)
        if maximum and images >= maximum:
            break
    return {"images": images,
            **{mode: {"median_coverage": float(np.median(values)),
                      "contained_pct": float(100 * np.mean(np.asarray(values) > .9)),
                      "children": len(values)} for mode, values in coverage.items()}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("action", choices=("greenness", "semantic", "instances", "nesting"))
    parser.add_argument("archive", type=Path)
    parser.add_argument("--images", type=int, default=400)
    parser.add_argument("--field", default="ground_truth")
    parser.add_argument("--mask-field", default="ground_truth_semantic")
    parser.add_argument("--fields", default="ground_truth")
    parser.add_argument("--child", default="leaf")
    parser.add_argument("--parent", default="whole plant")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    if args.action == "greenness":
        result = greenness(args.archive, args.images, args.field, args.seed)
    elif args.action == "semantic":
        result = semantic(args.archive, args.images, args.field, args.mask_field, args.seed)
    elif args.action == "instances":
        result = instances(args.archive, args.images, args.field, args.seed)
    else:
        result = nesting(args.archive, args.images, args.fields.split(","),
                         set(args.child.split(",")), set(args.parent.split(",")))
    payload = json.dumps(result, indent=2, sort_keys=True)
    if args.out:
        args.out.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
