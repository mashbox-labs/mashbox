#!/bin/sh
# The catalog's record guard with its records required (tests/cpu/test_catalog.py::test_supported_rows_have_records).
# The private merge and release gates run this. A missing sidecar (internal/catalog-evidence.json, or the file
# MASHBOX_CATALOG_EVIDENCE names) or records root fails here; plain pytest skips the test, because the records are kept
# outside the public tree. Exit status: pytest's (0 only when every row holds against its records).
cd "$(dirname "$0")/.." || exit 2
MASHBOX_REQUIRE_RECORDS=1 exec "${PYTHON:-python3}" -m pytest -q -p no:cacheprovider -rfs --tb=short \
    "tests/cpu/test_catalog.py::test_supported_rows_have_records"
