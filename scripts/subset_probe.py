"""Overflow counts for one archive split into groups, at several tolerances,
with the largest excess reported for both readings.

A box array whose stored numbers are corner coordinates clipped into the frame
can only leave the frame, under the centre reading, on the left and top; the
right and bottom counts are exactly zero by algebra.  Whether the corner reading
of the same array is inside the frame is the complementary check, and it is the
one whose answer depends on how a group's labels were written: some archives in
this study clip with an epsilon, so a strict ``> 1`` test reports thousands of
"overflows" that never exceed 5e-06 normalised units (0.005 px on a 1000-px
image) while a tolerant test reports none.  A real anchor displacement, by
contrast, is half a box and survives any tolerance a consumer would use.  Both
readings and both counts are printed per group, so the tolerance is a stated
decision rather than an implicit one.

Grouping is a decision too, and three are offered: by media sub-directory
(``path``, ``--depth`` components deep), by class label (``label``), or by
whether the detection carries a mask (``mask``), which separates a release's box
subset from its instance-mask subset when both live in one field.

Usage
-----
    python subset_probe.py ARCHIVE_DIR [--by path|label|mask] [--depth 2]
                                       [--tol 1e-5] [--out out.json]
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
SIDES = ("left", "top", "right", "bottom")
READS = ("centre", "corner")


class Counter:
    """Excesses beyond each frame side, kept separately for the two readings."""

    def __init__(self, tolerances):
        self.n = 0
        self.tolerances = tolerances
        self.excess = {read: defaultdict(list) for read in READS}

    def add(self, x, y, w, h):
        self.n += 1
        centre = (w / 2.0 - x, h / 2.0 - y, x + w / 2.0 - 1.0, y + h / 2.0 - 1.0)
        corner = (-x, -y, x + w - 1.0, y + h - 1.0)
        for read, spans in zip(READS, (centre, corner)):
            for side, amount in zip(SIDES, spans):
                if amount > 0.0:
                    self.excess[read][side].append(amount)

    def report(self):
        out = {"boxes": self.n}
        for read in READS:
            per_read = {}
            for side in SIDES:
                values = sorted(self.excess[read][side])
                cells = {
                    "worst_excess": values[-1] if values else 0.0,
                    "median_excess": values[len(values) // 2] if values else 0.0,
                }
                for t in self.tolerances:
                    cells["over_%g" % t] = sum(1 for v in values if v > t)
                per_read[side] = cells
            out[read] = per_read
        return out


def group_of(match, blob, mode, depth, key, index):
    if mode == "label":
        return match.group(1).decode("ascii")
    if mode == "mask":
        # a detection's own fields run to the start of the next detection
        nxt = BOX_RE.search(blob, match.end())
        window = blob[match.end():nxt.start()] if nxt else blob[match.end():]
        return "masked" if b'"mask"' in window else "boxes"
    offsets, names = index
    i = bisect_right(offsets, match.start()) - 1
    return names[i] if i >= 0 else "(root)"


def path_index(blob, key, depth):
    pattern = re.compile(('"%s": "([^"]*)"' % key).encode("ascii"))
    hits = list(pattern.finditer(blob))
    offsets, names = [], []
    for m in hits:
        parts = m.group(1).decode("ascii").split("/")
        offsets.append(m.start())
        names.append("/".join(parts[:depth]))
    return offsets, names


def scan(path, mode, depth, key, tolerances):
    with open(path, "rb") as handle:
        blob = handle.read()
    field_offsets = [m.start() for m in FIELD_RE.finditer(blob)]
    field_names = [m.group(1).decode("ascii") for m in FIELD_RE.finditer(blob)]
    index = path_index(blob, key, depth) if mode == "path" else None
    stats = defaultdict(lambda: Counter(tolerances))
    for match in BOX_RE.finditer(blob):
        x, y, w, h = (float(match.group(i)) for i in range(2, 6))
        i = bisect_right(field_offsets, match.start()) - 1
        field = field_names[i] if i >= 0 else "ground_truth"
        group = group_of(match, blob, mode, depth, key, index)
        stats[(group, field)].add(x, y, w, h)
    return stats


def line(group, field, read, rep, strict_key, tol_key):
    cells = []
    for side in SIDES:
        d = rep[read][side]
        cells.append("%s %7d/%-7d w=%.4g" % (side[:1], d[strict_key], d[tol_key],
                                             d["worst_excess"]))
    return "  %-26s %-16s %-7s %8d | %s" % (group[:26], field[:16], read,
                                            rep["boxes"], "  ".join(cells))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("directory")
    parser.add_argument("--by", default="path", choices=("path", "label", "mask"))
    parser.add_argument("--depth", type=int, default=1,
                        help="path components to group by, with --by path")
    parser.add_argument("--key", default="filepath")
    parser.add_argument("--tol", type=float, default=1e-5,
                        help="tolerance reported alongside the strict count")
    parser.add_argument("--meta", default=None,
                        help="metadata file name; both samples.json and "
                             "frames.json are used when omitted")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    tolerances = (0.0, 1e-9, 1e-6, args.tol, 1e-4, 1e-3)
    strict_key, tol_key = "over_%g" % 1e-9, "over_%g" % args.tol
    metas = [args.meta] if args.meta else ["samples.json", "frames.json"]
    payload = {}
    for meta in metas:
        path = os.path.join(args.directory, meta)
        if not os.path.exists(path):
            continue
        stats = scan(path, args.by, args.depth, args.key, tolerances)
        print("\n==== %s  (grouped by %s)" % (meta, args.by))
        print("  %-26s %-16s %-7s %8s | %s"
              % ("group", "field", "read", "boxes",
                 " ".join("%-22s" % s for s in SIDES)))
        report = {}
        for (group, field), counter in sorted(stats.items()):
            rep = counter.report()
            report["%s/%s" % (group, field)] = rep
            for read in READS:
                print(line(group, field, read, rep, strict_key, tol_key))
        print("  (cells are: side, boxes outside strictly / outside by more than "
              "%g, worst excess)" % args.tol)
        payload[meta] = report
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=1, sort_keys=True)
        print("\nwrote %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
