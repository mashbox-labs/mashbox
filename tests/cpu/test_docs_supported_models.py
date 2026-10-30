"""docs/supported-models.md: the catalog block is the render of models.json (mashbox.catalog, as `mashbox models`), the
page is in the docs sidebar, and the offline sync check reports nothing on it (its MASHBOX Labs publisher heading is
generated, not a description of the company)."""
import importlib.util
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAGE = ROOT / "docs" / "supported-models.md"
_spec = importlib.util.spec_from_file_location("sync_check", ROOT / "tools" / "hf_github_sync_check.py")
SC = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(SC)


def test_page_holds_the_catalog_block_and_is_in_the_sidebar():
    text = PAGE.read_text()
    assert text.count("<!-- catalog:start -->") == 1 and text.count("<!-- catalog:end -->") == 1
    assert "(supported-models.md)" in (ROOT / "docs" / "SUMMARY.md").read_text()


def test_offline_sync_check_has_no_problem_on_the_page():
    out = subprocess.run([sys.executable, "-I", str(ROOT / "tools" / "hf_github_sync_check.py"), "check", "--offline",
                          "--repo", str(ROOT)], capture_output=True, text=True, timeout=600)
    bad = [line for line in (out.stdout + out.stderr).splitlines() if "docs/supported-models.md" in line]
    assert bad == [], bad


def test_every_level_the_catalog_prints_is_defined_in_the_levels_table():
    p = SC.Problems()
    SC.check_levels(ROOT, p)
    assert p == [], p


def test_a_printed_level_missing_from_the_levels_table_is_named(tmp_path):
    """the level dropped from the Levels table is one the page's rows print (which levels they reach follows
    models.json, so the test takes the first GGUF level the page prints)"""
    text = PAGE.read_text()
    printed = [lv for lv in ("Tuned", "Tested", "Quick check", "Checked on CPU") if f"| {lv}<br>" in text]
    assert printed, "the page prints no GGUF level"
    lv = printed[0]
    assert f"| {lv} |" in text.split("## Levels", 1)[1]
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "supported-models.md").write_text(
        "\n".join(x for x in text.splitlines() if not x.startswith(f"| {lv} |")) + "\n")
    p = SC.Problems()
    SC.check_levels(tmp_path, p)
    assert len(p) == 1 and f"level '{lv}'" in p[0], p
