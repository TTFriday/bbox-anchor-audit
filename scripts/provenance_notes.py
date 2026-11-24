"""Collect the archive-side provenance statements used by the materials table.

Every public mirror ships a dataset card and a FiftyOne ``metadata.json``. This
script copies, for each archive, the fields the paper has to state verbatim:
the licence, the cited source paper and repository, the sample count and media
type, and any card sentence that describes how the four stored box numbers were
constructed. It writes UTF-8 to a file rather than stdout because the console
encoding on the audit machine cannot print the card text.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re

PATTERNS = re.compile(
    r"(license|licen[cs]e_name|paper:|repository:|homepage:|citation|doi|arxiv|"
    r"bounding.box|bbox|\[x,? ?y,? ?w,? ?h\]|find_objects|Label Studio|labelme|"
    r"cvat|yolo|converted|conversion|normalis|normaliz)", re.I)
SKIP = re.compile(r"^\s*$")


def card_lines(path):
    keep, seen = [], set()
    with io.open(path, "r", encoding="utf-8", errors="replace") as handle:
        for number, line in enumerate(handle, 1):
            if SKIP.match(line) or not PATTERNS.search(line):
                continue
            text = line.strip()
            if len(text) > 400:
                text = text[:400] + " ..."
            if text in seen:
                continue
            seen.add(text)
            keep.append("%4d  %s" % (number, text))
    return keep


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("root", help="directory containing one sub-directory per archive")
    parser.add_argument("--out", default="provenance.txt")
    args = parser.parse_args(argv)
    out = io.open(args.out, "w", encoding="utf-8")
    for name in sorted(os.listdir(args.root)):
        directory = os.path.join(args.root, name)
        if not os.path.isdir(directory) or name.startswith(("_", ".")):
            continue
        out.write("\n" + "=" * 78 + "\n%s\n" % name)
        meta = os.path.join(directory, "metadata.json")
        if os.path.exists(meta):
            with io.open(meta, "r", encoding="utf-8", errors="replace") as handle:
                blob = json.load(handle)
            summary = {k: v for k, v in blob.items()
                       if k in ("name", "media_type", "num_samples", "persistent",
                                "local_dir")}
            out.write("metadata.json: %s\n" % json.dumps(summary, sort_keys=True))
            for key in ("dataset_info", "info"):
                info = blob.get(key)
                if isinstance(info, dict):
                    for ik in sorted(info):
                        value = str(info[ik])
                        out.write("  %s.%s = %s\n"
                                  % (key, ik, value[:220] + (" ..." if len(value) > 220 else "")))
        for card in ("README.md", "README.MD", "readme.md"):
            path = os.path.join(directory, card)
            if os.path.exists(path):
                out.write("%s:\n" % card)
                for line in card_lines(path):
                    out.write("  " + line + "\n")
                break
        yml = os.path.join(directory, "fiftyone.yml")
        if os.path.exists(yml):
            with io.open(yml, "r", encoding="utf-8", errors="replace") as handle:
                out.write("fiftyone.yml: %s\n" % handle.read().strip().replace("\n", " | ")[:300])
    out.close()
    print("wrote %s" % args.out)


if __name__ == "__main__":
    main()
