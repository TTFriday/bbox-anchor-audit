"""Enumerate the ways a four-number box array can be misread, and ask which
misreadings leave the IoU between the two readings constant.

Four formats are in ordinary use for a normalised array (a, b, c, d):

=======  ===================================================
name     box it denotes
=======  ===================================================
centre   ``[a - c/2, b - d/2, a + c/2, b + d/2]``   (xc, yc, w, h)
corner   ``[a, b, a + c, b + d]``                   (x,  y,  w, h)
half     ``[a - c, b - d, a + c, b + d]``           (xc, yc, w/2, h/2)
corners  ``[a, b, c, d]``                           (x0, y0, x1, y1)
=======  ===================================================

For every ordered pair (stored as, read as) the script sweeps a grid of widths,
heights and positions and reports the minimum and maximum IoU between the two
readings.  A pair whose IoU has min == max is scale-free: its value alone
identifies the misreading, which is what makes it usable as a diagnostic.

    python convention_enumeration.py [--out convention_enumeration.json]
"""

from __future__ import annotations

import argparse
import itertools
import json

FORMATS = ("centre", "corner", "half", "corners")


def box(fmt, a, b, c, d):
    if fmt == "centre":
        return (a - c / 2.0, b - d / 2.0, a + c / 2.0, b + d / 2.0)
    if fmt == "corner":
        return (a, b, a + c, b + d)
    if fmt == "half":
        return (a - c, b - d, a + c, b + d)
    if fmt == "corners":
        return (a, b, c, d)
    raise ValueError(fmt)


def iou(p, q):
    width = min(p[2], q[2]) - max(p[0], q[0])
    height = min(p[3], q[3]) - max(p[1], q[1])
    if width <= 0 or height <= 0:
        return 0.0
    inter = width * height
    area_p = (p[2] - p[0]) * (p[3] - p[1])
    area_q = (q[2] - q[0]) * (q[3] - q[1])
    union = area_p + area_q - inter
    return inter / union if union > 0 else 0.0


def stored(fmt, x, y, w, h):
    """The four numbers a file holds when its true convention is *fmt*."""
    if fmt == "corners":
        return (x, y, x + w, y + h)
    if fmt == "corner":
        return (x, y, w, h)
    if fmt == "centre":
        return (x + w / 2.0, y + h / 2.0, w, h)
    if fmt == "half":
        return (x + w / 2.0, y + h / 2.0, w / 2.0, h / 2.0)
    raise ValueError(fmt)


def grid():
    widths = [1.0, 2.0, 7.0, 33.0, 100.0, 480.0, 1080.0]
    ratios = [0.05, 0.2, 1.0, 5.0, 20.0]
    positions = [(0.0, 0.0), (17.0, 3.0), (500.0, 250.0), (0.5, 0.5)]
    for w, r, (x, y) in itertools.product(widths, ratios, positions):
        yield x, y, w, w * r


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="convention_enumeration.json")
    args = parser.parse_args()

    rows = []
    print("%-9s %-9s %-11s %-12s %s" % ("stored", "read", "invariant", "value", "note"))
    for stored_fmt, read_fmt in itertools.permutations(FORMATS, 2):
        values = []
        for x, y, w, h in grid():
            a, b, c, d = stored(stored_fmt, x, y, w, h)
            values.append(iou(box(stored_fmt, a, b, c, d), box(read_fmt, a, b, c, d)))
        low, high = min(values), max(values)
        invariant = high - low < 1e-12
        note = ""
        if invariant:
            note = {7.0: "1/7", 4.0: "1/4"}.get(round(1.0 / low) if low > 0 else 0, "")
            if low == 0.0:
                note = "0 (disjoint)"
        print("%-9s %-9s %-11s %-12s %s" % (
            stored_fmt, read_fmt, "yes" if invariant else "no",
            ("%.6f" % low) if invariant else "[%.4f, %.4f]" % (low, high), note))
        rows.append({"stored_as": stored_fmt, "read_as": read_fmt,
                     "invariant": invariant, "min": low, "max": high,
                     "value": low if invariant else None,
                     "one_over": (1.0 / low if invariant and low > 0 else None)})
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=1)
    print("\n%d pairs, %d of them invariant -> %s"
          % (len(rows), sum(r["invariant"] for r in rows), args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
