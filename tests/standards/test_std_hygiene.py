"""Release hygiene rows of the standards tier that need no tiny engine.

ST-H6  two first loads at once: GGUF files that share one tokenizer (the same tokenizer files, so the same folder under
       MASHBOX_CACHE/gguf) loaded by two writers on a fresh cache, the second one running exactly between the first one's
       write of a file and its rename into place (as two `mashbox run` processes started together can). Both get the
       folder, and every file in it is whole. The interleaving is forced (a hook on os.replace), so the result does not
       depend on timing or core count.
"""
import json
import os
import threading

import pytest

import tiny

pytestmark = pytest.mark.standards


@pytest.mark.xfail(strict=True, reason="open: src/mashbox/gguf/tokenizer.py model_dir writes every file through the same "
                                       "'<name>.part' path, so the first writer's rename finds its .part already moved "
                                       "by the second writer (FileNotFoundError)")
def test_h6_two_first_loads_share_the_tokenizer_folder(tmp_path, monkeypatch):
    from mashbox.gguf import tokenizer as GT
    from mashbox.gguf.reader import File
    fams = [n for n in tiny.families() if n != tiny.BSHJ]
    assert fams, "no GGUF family in the fixture table"
    p, root = tiny.weights(fams[0], "random"), str(tmp_path / "cache")
    real, st, errs, dirs = os.replace, dict(fired=False), [], []

    def second():
        try:
            dirs.append(GT.model_dir(File(p), p, root=root))
        except Exception as e:          # noqa: BLE001 (reported below)
            errs.append(f"second writer: {type(e).__name__}: {e}")

    def replace(a, b, *args, **kw):
        if not st["fired"] and str(a).endswith(".part"):
            st["fired"] = True
            t = threading.Thread(target=second)
            t.start()
            t.join(60)
        return real(a, b, *args, **kw)

    monkeypatch.setattr(os, "replace", replace)
    try:
        dirs.append(GT.model_dir(File(p), p, root=root))
    except Exception as e:              # noqa: BLE001
        errs.append(f"first writer: {type(e).__name__}: {e}")
    monkeypatch.setattr(os, "replace", real)
    assert st["fired"], "the second writer never ran between a write and its rename: the check proves nothing"
    assert not errs, errs
    assert len(set(dirs)) == 1 and len(dirs) == 2, dirs
    d = dirs[0]
    for k in ("tokenizer.json", "generation_config.json"):
        with open(os.path.join(d, k), encoding="utf-8") as fh:
            json.load(fh)
    with open(os.path.join(d, "chat_template.jinja"), encoding="utf-8") as fh:
        assert fh.read().strip(), "empty chat template"
    assert not [n for n in os.listdir(d) if n.endswith(".part")], os.listdir(d)
