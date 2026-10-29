"""docs/supported-models.md: the catalog block is the render of models.json (mashbox.catalog, as `mashbox models`), the
page is in the docs sidebar, and the offline sync check reports nothing on it (its MASHBOX Labs publisher heading is
generated, not a description of the company)."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAGE = ROOT / "docs" / "supported-models.md"


def test_page_holds_the_catalog_block_and_is_in_the_sidebar():
    text = PAGE.read_text()
    assert text.count("<!-- catalog:start -->") == 1 and text.count("<!-- catalog:end -->") == 1
    assert "(supported-models.md)" in (ROOT / "docs" / "SUMMARY.md").read_text()


def test_offline_sync_check_has_no_problem_on_the_page():
    out = subprocess.run([sys.executable, "-I", str(ROOT / "tools" / "hf_github_sync_check.py"), "check", "--offline",
                          "--repo", str(ROOT)], capture_output=True, text=True, timeout=600)
    bad = [line for line in (out.stdout + out.stderr).splitlines() if "docs/supported-models.md" in line]
    assert bad == [], bad
