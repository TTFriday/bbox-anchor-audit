"""Do the dimensions declared in the label metadata match the media on disk?

The extent-plausibility statistic of ``fingerprint.py`` multiplies a stored
normalised width and height by the ``metadata.width`` / ``metadata.height`` that
the archive declares for each sample.  That is only meaningful if the declared
dimensions are the dimensions the annotation tool worked in.  Camera JPEGs can
carry an EXIF orientation tag, and a tool that reads the raw decoded size while
the annotator, or a viewer, sees a transposed image yields an aspect ratio that
is turned through 90 degrees rather than corrupted.  This script reads image
headers only -- raw size, EXIF orientation tag, and the size after the transpose
-- for every readable sample of each group, and compares them with what the
metadata claims, so an implausible aspect distribution can be attributed either
to the stored extents or to a rotation convention.

Usage
-----
    python declared_size_check.py ARCHIVE_DIR [--key site] [--limit 0]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict

from PIL import Image

ORIENTATION_TAG = 274


def records(path, key):
    """Yield (filepath, (declared W, declared H), group) per sample record."""
    with open(path, "rb") as handle:
        blob = handle.read()
    hits = list(re.finditer(rb'"filepath": "([^"]*)"', blob))
    sizes = [(int(m.group(1)), int(m.group(2)))
             for m in re.finditer(rb'"width": (\d+), "height": (\d+)', blob)]
    groups = ([m.group(1).decode("ascii", "replace")
               for m in re.finditer(('"%s": "([^"]*)"' % key).encode("ascii"), blob)]
              if key else [])
    for i, match in enumerate(hits):
        yield (match.group(1).decode("ascii", "replace"),
               sizes[i] if i < len(sizes) else None,
               groups[i] if i < len(groups) else "(all)")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("directory")
    parser.add_argument("--meta", default="samples.json")
    parser.add_argument("--key", default="site",
                        help="sample-level attribute to group by, or none")
    parser.add_argument("--limit", type=int, default=0,
                        help="stop after this many readable samples per group")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    by_group = defaultdict(list)
    for filepath, size, group in records(os.path.join(args.directory, args.meta),
                                         args.key):
        if len(by_group[group]) < args.limit or not args.limit:
            by_group[group].append((filepath, size))

    summary = {}
    for group, items in sorted(by_group.items()):
        seen = defaultdict(int)
        declared, raw = [], []
        for filepath, size in items:
            disk = os.path.join(args.directory, filepath.replace("/", os.sep))
            if not os.path.exists(disk):
                seen["media_absent"] += 1
                continue
            try:
                with Image.open(disk) as handle:
                    # Header reads only: ``size`` and ``getexif`` do not decode
                    # the pixel data, which matters on multi-megapixel JPEGs.
                    here = handle.size
                    orientation = handle.getexif().get(ORIENTATION_TAG, "none")
            except Exception as exc:                        # noqa: BLE001
                seen["unreadable"] += 1
                seen["error:%s" % type(exc).__name__] += 1
                continue
            seen["readable"] += 1
            seen["orientation:%s" % orientation] += 1
            # Orientation values 5-8 rotate by a quarter turn, so a viewer that
            # honours EXIF shows a frame transposed against the raw size.
            transposes = isinstance(orientation, int) and 5 <= orientation <= 8
            shown = (here[1], here[0]) if transposes else here
            seen["transpose_changes_size:%s" % (here != shown)] += 1
            if size is None:
                seen["declared_missing"] += 1
            else:
                seen["declared_equals_raw:%s" % (tuple(size) == here)] += 1
                seen["declared_equals_shown:%s" % (tuple(size) == shown)] += 1
                declared.append(size[0] / float(size[1]))
            raw.append(here[0] / float(here[1]))
        def median(values):
            return sorted(values)[len(values) // 2] if values else None
        summary[group] = {"samples": len(items), "counters": dict(seen),
                          "median_declared_aspect": median(declared),
                          "median_raw_aspect": median(raw)}
        print("\n%-16s %s" % (group, json.dumps(summary[group], sort_keys=True)))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=1, sort_keys=True)
        print("\nwrote %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
