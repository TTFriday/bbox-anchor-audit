# Bounding-box anchor audit

Label-side audit code and aggregate results for bounding-box convention
errors in public plant-imaging datasets.  The audit compares the documented
FiftyOne centre interpretation `[xc, yc, w, h]` with the top-left
interpretation `[x, y, w, h]` without rewriting any source annotation.

## Contents

- `scripts/fingerprint.py`: metadata-only S1--S4 diagnostics.
- `scripts/convention_enumeration.py`: algebraic IoU enumeration.
- `scripts/subset_probe.py`: overflow counts by path, class or mask subset.
- `scripts/declared_size_check.py`: image-header and EXIF size check;
  read-only unless `--out` is supplied.
- `scripts/provenance_notes.py`: extract source/licence statements.
- `scripts/media_confirmation.py`: clean read-only pixel/mask checks.
- `data/`: aggregate JSON, class counts and extracted provenance.
- `results/logs/`: historical full-population fingerprint logs and capped
  media-confirmation logs.
- `tests/test_geometry.py`: synthetic geometry checks.

## Run the audit

```bash
python scripts/fingerprint.py ARCHIVE_ROOT --out fingerprint.json
python scripts/subset_probe.py ARCHIVE_DIR --by mask --out subset.json
python scripts/declared_size_check.py ARCHIVE_DIR --out size_check.json
python -m unittest discover -s tests
```

`ARCHIVE_ROOT` contains one FiftyOne archive per sub-directory.  Metadata
scripts do not require the FiftyOne Python package.  Media checks require
the original archive media and are read-only by default.

## Interpretation boundaries

FiftyOne documents centre-format boxes.  Several audited exports contain
top-left-format arrays despite that documentation, which can shift a box by
half its width and height when consumed as centre-format.  Five archives
support the top-left interpretation; Brandenburg remains unresolved.

The centre/corner IoU is algebraically `1/7` for every positive-area box.
It cannot identify the stored convention.  One-sided overflow, boundary
mass, declared image size and pixel/mask registration are complementary
diagnostics, not an automatic conversion rule.  Historical media checks
are capped samples and are never pooled with the full-population metadata
fingerprint.

The clean `media_confirmation.py` implementation reproduces the diagnostic
families, not necessarily the exact historical draws.  The archived capped
values in `data/confirmation_paths.json` remain the record of those runs.

No code licence is granted by this repository.  Dataset licences are listed
in `DATA_SOURCES.md`; no dataset files are redistributed here.
