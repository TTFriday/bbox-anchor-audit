"""Label-side anchor-convention and geometry audit for FiftyOne archives.

FiftyOne documents ``Detection.bounding_box`` as
``[x_center, y_center, width, height]``.  Exported archives can nevertheless
contain arrays written under another convention.  This program therefore reads
both the documented centre interpretation and the top-left ``[x, y, w, h]``
interpretation.  It reads only ``samples.json`` / ``frames.json`` metadata,
never media, and reports four diagnostic statistics.

S1 compares out-of-frame counts, directional asymmetry and the IoU between the
two readings.  For positive-area boxes the centre/corner IoU is algebraically
1/7, so that value cannot identify the stored convention.  One-sided overflow
is evidence consistent with a corner array clipped by a centre-reading tool;
two-sided overflow can also arise from source annotations that were never
clipped and is not by itself a definitive format decision.

S2 reports the pixel-space aspect distribution implied by declared sample
width and height.  S3 reports boundary mass for stored components and derived
edges.  S4 reports the largest excess beyond each frame side, with strict and
tolerance-based counts.  These are diagnostics, not an automatic convention
conversion or a safe abstention policy.

Usage::

    python fingerprint.py ARCHIVE_ROOT [--out fingerprint.json]
                                       [--group site] [--tol 1e-5] [name ...]

``ARCHIVE_ROOT`` contains one sub-directory per archive.  Optional names limit
the run.  ``--group`` additionally splits the report by a sample attribute.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from bisect import bisect_right
from collections import defaultdict

BOX_RE = re.compile(
    rb'"label": "([^"]*)", "bounding_box": \[([-\d.eE+]+), ([-\d.eE+]+), '
    rb'([-\d.eE+]+), ([-\d.eE+]+)\]')
FIELD_RE = re.compile(rb'"([A-Za-z][A-Za-z0-9_]*)": \{"_cls": "Detections"')
SIZE_RE = re.compile(rb'"width": (\d+), "height": (\d+)')
ONE_SEVENTH = 1.0 / 7.0
EPS = 1e-9
TOL = 1e-5
SIDES = ("left", "right", "top", "bottom")
QUANTILES = (0.01, 0.05, 0.10, 0.50, 0.90, 0.95, 0.99)


def quantiles(values, probs=QUANTILES):
    if not values:
        return {}
    ordered = sorted(values)
    n = len(ordered)
    return {"p%02d" % round(p * 100): ordered[min(n - 1, max(0, round(p * (n - 1))))]
            for p in probs}


def centre_read(x, y, w, h):
    return (x - w / 2.0, y - h / 2.0, x + w / 2.0, y + h / 2.0)


def corner_read(x, y, w, h):
    return (x, y, x + w, y + h)


def overlap_over_union(p, q):
    width = min(p[2], q[2]) - max(p[0], q[0])
    height = min(p[3], q[3]) - max(p[1], q[1])
    if width <= 0 or height <= 0:
        return 0.0
    inter = width * height
    area_p = (p[2] - p[0]) * (p[3] - p[1])
    area_q = (q[2] - q[0]) * (q[3] - q[1])
    union = area_p + area_q - inter
    return inter / union if union > 0 else 0.0


def excesses(box):
    """How far each of the four frame sides is exceeded, 0 when it is not.

    ``box`` holds (left, top, right, bottom) in normalised image units; the
    return value is ordered like ``SIDES``.
    """
    return (-box[0], box[2] - 1.0, -box[1], box[3] - 1.0)


class FieldStats:
    def __init__(self):
        self.n = 0
        self.centre_strict = [0, 0, 0, 0]
        self.centre_tol = [0, 0, 0, 0]
        self.corner_strict = [0, 0, 0, 0]
        self.corner_tol = [0, 0, 0, 0]
        self.centre_worst = [0.0, 0.0, 0.0, 0.0]
        self.corner_worst = [0.0, 0.0, 0.0, 0.0]
        self.negative = 0
        self.negative_beyond_tol = 0
        self.degenerate = 0
        self.iou_min = None
        self.iou_max = None
        self.iou_off = 0
        self.iou_all = []
        self.first = []
        self.aspect_px = []
        self.aspect_norm = []
        self.area = []
        self.labels = defaultdict(int)
        self.at_boundary = defaultdict(int)
        self.at_boundary_exact = defaultdict(int)
        self.lo = {}
        self.hi = {}

    def _span(self, name, value):
        if name not in self.lo or value < self.lo[name]:
            self.lo[name] = value
        if name not in self.hi or value > self.hi[name]:
            self.hi[name] = value

    def add(self, label, x, y, w, h, image=None):
        self.n += 1
        self.labels[label] += 1
        if min(x, y, w, h) < -EPS:
            self.negative += 1
        if min(x, y, w, h) < -TOL:
            self.negative_beyond_tol += 1
        if w <= EPS or h <= EPS:
            self.degenerate += 1
        for strict, beyond, worst, box in (
                (self.centre_strict, self.centre_tol, self.centre_worst,
                 centre_read(x, y, w, h)),
                (self.corner_strict, self.corner_tol, self.corner_worst,
                 corner_read(x, y, w, h))):
            for i, amount in enumerate(excesses(box)):
                if amount > EPS:
                    strict[i] += 1
                if amount > TOL:
                    beyond[i] += 1
                if amount > worst[i]:
                    worst[i] = amount
        for name, value in (("first_component", x), ("second_component", y),
                            ("extent_w", w), ("extent_h", h),
                            ("corner_right_edge", x + w), ("corner_bottom_edge", y + h),
                            ("centre_right_edge", x + w / 2.0),
                            ("centre_bottom_edge", y + h / 2.0),
                            ("centre_left_edge", x - w / 2.0),
                            ("centre_top_edge", y - h / 2.0)):
            self._span(name, value)
            if abs(value) <= TOL or abs(value - 1.0) <= TOL:
                self.at_boundary[name] += 1
            if value == 0.0 or value == 1.0:
                self.at_boundary_exact[name] += 1
        value = overlap_over_union(corner_read(x, y, w, h), centre_read(x, y, w, h))
        self.iou_min = value if self.iou_min is None else min(self.iou_min, value)
        self.iou_max = value if self.iou_max is None else max(self.iou_max, value)
        if abs(value - ONE_SEVENTH) > 1e-9:
            self.iou_off += 1
        self.iou_all.append(value)
        self.first.append(x)
        self.area.append(w * h)
        if h > EPS:
            self.aspect_norm.append(w / h)
            if image:
                self.aspect_px.append((w * image[0]) / (h * image[1]))

    def report(self):
        n = max(self.n, 1)

        def block(strict, beyond, worst):
            out = {}
            for i, side in enumerate(SIDES):
                out[side] = strict[i]
                out["%s_strict_pct" % side] = round(100.0 * strict[i] / n, 4)
                out["%s_count" % side] = beyond[i]
                out["%s_pct" % side] = round(100.0 * beyond[i] / n, 4)
                out["%s_worst_excess" % side] = round(worst[i], 6)
            return out

        def boundary(name):
            return {
                "at_boundary_pct": round(100.0 * self.at_boundary[name] / n, 4),
                "exact_pct": round(100.0 * self.at_boundary_exact[name] / n, 4),
                "min": self.lo.get(name), "max": self.hi.get(name),
            }

        # S1's oriented reading of the overflow counts.  A side whose count is
        # zero in a given reading cannot have been produced by an array clipped
        # in the other format, so a two-sided reading is evidence that that
        # reading is the stored format.
        def two_sided(beyond):
            return ((beyond[0] > 0 and beyond[1] > 0)
                    or (beyond[2] > 0 and beyond[3] > 0))

        centre_two_sided = two_sided(self.centre_tol)
        corner_two_sided = two_sided(self.corner_tol)
        orientation = ("centre" if centre_two_sided and not corner_two_sided else
                       "corner" if corner_two_sided and not centre_two_sided else
                       "both" if centre_two_sided and corner_two_sided else "neither")
        fingerprint = (self.centre_strict[1] == 0 and self.centre_strict[3] == 0
                       and max(self.centre_strict[0], self.centre_strict[2]) > 0)
        names = ("first_component", "second_component", "corner_right_edge",
                 "corner_bottom_edge", "centre_right_edge", "centre_bottom_edge",
                 "extent_w", "extent_h")
        return {
            "boxes": self.n,
            "classes": len(self.labels),
            "top_classes": dict(sorted(self.labels.items(),
                                       key=lambda kv: -kv[1])[:8]),
            "centre_read_out_of_frame": block(self.centre_strict, self.centre_tol,
                                              self.centre_worst),
            "corner_read_out_of_frame": block(self.corner_strict, self.corner_tol,
                                              self.corner_worst),
            "one_sided_fingerprint": fingerprint,
            "s1_orientation": orientation,
            "asymmetry_x": round(abs(self.centre_strict[0] - self.centre_strict[1])
                                 / (self.centre_strict[0] + self.centre_strict[1] + 1), 6),
            "asymmetry_y": round(abs(self.centre_strict[2] - self.centre_strict[3])
                                 / (self.centre_strict[2] + self.centre_strict[3] + 1), 6),
            "stored_negative_components": self.negative,
            "stored_negative_beyond_tol": self.negative_beyond_tol,
            "degenerate_boxes": self.degenerate,
            "boundary_clipping": {name: boundary(name) for name in names},
            "iou_corner_vs_centre": {
                "min": self.iou_min, "max": self.iou_max,
                "median": sorted(self.iou_all)[len(self.iou_all) // 2],
                "not_one_seventh": self.iou_off,
                "not_one_seventh_pct": round(100.0 * self.iou_off / n, 4),
            },
            "first_component": quantiles(self.first),
            "aspect_w_over_h_normalised": quantiles(self.aspect_norm,
                                                    (0.05, 0.50, 0.95)),
            "aspect_pixels": quantiles(self.aspect_px, (0.05, 0.25, 0.50, 0.75, 0.95)),
            "aspect_pixels_implausible_pct": (
                round(100.0 * sum(1 for v in self.aspect_px if not 0.33 <= v <= 3.0)
                      / len(self.aspect_px), 3) if self.aspect_px else None),
            "area_wh": quantiles(self.area, (0.05, 0.50, 0.95)),
        }


def markers(regex, blob):
    found = list(regex.finditer(blob))
    return [m.start() for m in found], found


def scan(path, group_key=None):
    """Return {field (or group/field): FieldStats} for a metadata file."""
    with open(path, "rb") as handle:
        blob = handle.read()
    field_offsets, field_hits = markers(FIELD_RE, blob)
    field_names = [m.group(1).decode("ascii", "replace") for m in field_hits]
    size_offsets, size_hits = markers(SIZE_RE, blob)
    sizes = [(int(m.group(1)), int(m.group(2))) for m in size_hits]
    if group_key:
        pattern = re.compile(('"%s": "([^"]*)"' % group_key).encode("ascii"))
        group_offsets, group_hits = markers(pattern, blob)
        group_names = [m.group(1).decode("ascii", "replace") for m in group_hits]
    else:
        group_offsets, group_names = [], []
    stats = defaultdict(FieldStats)
    for match in BOX_RE.finditer(blob):
        try:
            x, y, w, h = (float(match.group(i)) for i in range(2, 6))
        except ValueError:
            continue
        start = match.start()
        i = bisect_right(field_offsets, start) - 1
        field = field_names[i] if i >= 0 else "ground_truth"
        j = bisect_right(size_offsets, start) - 1
        image = sizes[j] if j >= 0 else None
        k = bisect_right(group_offsets, start) - 1
        key = "%s/%s" % (group_names[k], field) if k >= 0 else field
        stats[key].add(match.group(1).decode("utf-8", "replace"), x, y, w, h, image)
    return stats, blob.count(b'"bounding_box": [')


def show(name, meta, key, rep):
    cen, cor, iou = (rep["centre_read_out_of_frame"],
                     rep["corner_read_out_of_frame"], rep["iou_corner_vs_centre"])
    clip = rep["boundary_clipping"]

    def pct(block, side):
        return "%s=%.2f%%(%.3f)" % (side[:1].upper(), block["%s_pct" % side],
                                    block["%s_worst_excess" % side])

    print("\n%-28s %-13s %s" % (name, meta, key))
    print("  n=%8d  classes=%d  one-sided=%s  S1 orientation=%s"
          % (rep["boxes"], rep["classes"], rep["one_sided_fingerprint"],
             rep["s1_orientation"]))
    print("  centre read  " + "  ".join(pct(cen, s) for s in SIDES)
          + "   strict L/R/T/B=%d/%d/%d/%d"
          % (cen["left"], cen["right"], cen["top"], cen["bottom"]))
    print("  corner read  " + "  ".join(pct(cor, s) for s in SIDES)
          + "   strict L/R/T/B=%d/%d/%d/%d"
          % (cor["left"], cor["right"], cor["top"], cor["bottom"]))
    print("  IoU[min,med,max]=[%.6f, %.6f, %.6f]  not 1/7 = %.4f%%"
          % (iou["min"], iou["median"], iou["max"], iou["not_one_seventh_pct"]))
    print("  boundary mass%%  x=%.2f y=%.2f | cornerR=%.2f cornerB=%.2f "
          "(exact %.2f/%.2f) | centreR=%.2f centreB=%.2f | w=%.2f h=%.2f"
          % (clip["first_component"]["at_boundary_pct"],
             clip["second_component"]["at_boundary_pct"],
             clip["corner_right_edge"]["at_boundary_pct"],
             clip["corner_bottom_edge"]["at_boundary_pct"],
             clip["corner_right_edge"]["exact_pct"],
             clip["corner_bottom_edge"]["exact_pct"],
             clip["centre_right_edge"]["at_boundary_pct"],
             clip["centre_bottom_edge"]["at_boundary_pct"],
             clip["extent_w"]["at_boundary_pct"], clip["extent_h"]["at_boundary_pct"]))
    print("  components     min x=%.6g y=%.6g  max cornerR=%.6g cornerB=%.6g  "
          "neg=%d (beyond tol %d) deg=%d"
          % (clip["first_component"]["min"], clip["second_component"]["min"],
             clip["corner_right_edge"]["max"], clip["corner_bottom_edge"]["max"],
             rep["stored_negative_components"], rep["stored_negative_beyond_tol"],
             rep["degenerate_boxes"]))
    print("  pixel aspect   p05/p50/p95=%s  implausible=%s%%  area p50=%.6g"
          % ([round(rep["aspect_pixels"].get(k, float("nan")), 3)
              for k in ("p05", "p50", "p95")],
             rep["aspect_pixels_implausible_pct"],
             rep["area_wh"].get("p50", float("nan"))))


def main(argv=None):
    global TOL
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("root", help="directory containing one sub-directory per archive")
    parser.add_argument("--out", default="fingerprint.json")
    parser.add_argument("--group", default=None,
                        help="split by this sample-level attribute, e.g. site")
    parser.add_argument("--tol", type=float, default=TOL,
                        help="normalised tolerance below which an excess is a clip "
                             "residual rather than an out-of-frame box")
    parser.add_argument("names", nargs="*")
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    TOL = args.tol

    names = args.names or sorted(
        d for d in os.listdir(args.root) if os.path.isdir(os.path.join(args.root, d)))
    result = {"_run": {"tolerance": TOL, "group": args.group,
                         "metadata_files": ["samples.json", "frames.json"],
                         "media_opened": False}}
    for name in names:
        directory = os.path.join(args.root, name)
        per_file = {}
        for meta in ("samples.json", "frames.json"):
            path = os.path.join(directory, meta)
            if not os.path.exists(path):
                continue
            stats, arrays = scan(path, args.group)
            parsed = sum(obj.n for obj in stats.values())
            print("\n### %s  %s  arrays=%d  parsed=%d  tol=%g"
                  % (name, meta, arrays, parsed, TOL))
            if parsed != arrays:
                print("    note: %d box arrays were not adjacent to a label key"
                      % (arrays - parsed))
            report = {}
            for key, obj in sorted(stats.items()):
                rep = obj.report()
                report[key] = rep
                show(name, meta, key, rep)
            if report:
                per_file[meta] = report
        if per_file:
            result[name] = per_file
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=1, sort_keys=True)
    print("\nwrote %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
