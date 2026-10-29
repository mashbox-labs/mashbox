"""The MASHBOX Chat module: the page meets the shell standard, resolves any model from models.json or the server, is
served by `mashbox run <model> --ui` and by `python -m mashbox.ui.chat --api URL`, and its helpers (Markdown, stream
reader, request body, timing) behave. CPU only, against tests/cpu/chat_stub.py. The browser run of the same page is
tests/cpu/chat_ui_e2e.py."""
import http.client
import json
import pathlib
import shutil
import socket
import subprocess
import sys
import threading
import time

import pytest

from mashbox.ui import chat, shell

ROOT = pathlib.Path(__file__).resolve().parents[2]
PAGE = ROOT / "src/mashbox/ui/chat/index.html"
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import chat_stub  # noqa: E402


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def get(port, path, method="GET", body=None):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    c.request(method, path, body=json.dumps(body) if body is not None else None, headers={"Content-Type": "application/json"})
    r = c.getresponse()
    return r.status, r.getheader("Content-Type"), r.read()


def blocks(html):
    import re
    return [json.loads(m) for m in re.findall(r'<script type="application/json" id="mbx-(?:module|chat)">(.*?)</script>', html, re.S)]


def test_page_meets_the_shell_standard():
    assert PAGE in shell.module_pages(ROOT)
    assert shell.check_page(PAGE.read_text()) == []
    html = PAGE.read_text()
    assert 'src="ui/chat/chat.js"' in html and "http://" not in html.replace("http://127.0.0.1", "")


def test_page_for_a_model_in_models_json():
    cfg, ch = blocks(chat.page("qwen3-8-27b").decode())
    assert (cfg["name"], cfg["module"], cfg["maker"]) == ("Qwen3.8-27B", "Chat", "Qwen")
    assert cfg["company"] == shell.COMPANY_LINE and cfg["links"]["docs"].endswith("/docs/models/qwen3-8-27b")
    assert cfg["resources"][0]["url"] == "https://huggingface.co/Qwen/Qwen3.8-27B"
    assert ch["thinking"] is True and set(ch["sampling"]) == {"thinking", "plain"}
    assert ch["sampling"]["plain"] == {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5, "repetition_penalty": 1.0}
    assert ch["source"].startswith("https://huggingface.co/Qwen/Qwen3.8-27B/blob/" + json.loads((ROOT / "models.json").read_text())["models"][1]["upstream"]["revision"]) or "Qwen3.8-27B" in ch["source"]


def test_page_for_any_other_model():
    cfg, ch = blocks(chat.page("someone/other-7b").decode())
    assert cfg["name"] == "someone/other-7b" and cfg["module"] == "Chat" and "docs" not in cfg["links"] and ch == {}


def test_chat_rows_in_models_json():
    reg = shell.registry()
    for m in reg["models"]:
        c = chat.chat_config(m["id"], reg)
        for mode in (c.get("sampling") or {}).values():
            assert set(mode) <= set(["temperature", "top_p", "top_k", "min_p", "presence_penalty", "repetition_penalty", "frequency_penalty"])
    assert (ROOT / "models.json").read_bytes() == (ROOT / "src/mashbox/models.json").read_bytes()


def test_routes():
    assert chat.handle("/", "qwen3-8-27b")[1].startswith("text/html")
    assert chat.handle("/chat", "qwen3-8-27b")[1].startswith("text/html")
    assert chat.handle("/ui/chat/chat.js", None)[1].startswith("text/javascript")
    assert chat.handle("/ui/shell/shell.css", None)[1].startswith("text/css")
    for bad in ("/chat/", "/ui/chat/../__init__.py", "/ui/chat/__init__.py", "/ui/chat/index.html", "/ui/chat/", "/v1/models", "/health", "/ui/shell/../chat/chat.js"):
        assert chat.handle(bad, None) is None, bad


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_js_helpers():
    r = subprocess.run(["node", str(ROOT / "tests/cpu/chat_js_checks.js"), str(ROOT / "src/mashbox/ui/chat/chat.js")], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0 and r.stdout.startswith("ok "), r.stdout + r.stderr


class FakeServer:      # what runtime.server.serve reads from a loaded model, without one
    model_id, ready = "qwen3-8-27b", True

    def models(self):
        return {"object": "list", "data": [{"id": self.model_id, "object": "model", "max_model_len": 2048}]}

    def log(self, m):
        pass


@pytest.mark.parametrize("ui", [True, False])
def test_model_server_serves_the_page_with_ui(ui):
    from mashbox.runtime import server
    port = free_port()
    threading.Thread(target=server.serve, args=(FakeServer(), "127.0.0.1", port, lambda m: None), kwargs={"ui": ui}, daemon=True).start()
    for _ in range(50):
        try:
            code, typ, body = get(port, "/")
            break
        except OSError:
            time.sleep(0.05)
    if ui:
        assert code == 200 and typ.startswith("text/html") and b'id="mbx-chat"' in body and b"Qwen3.8-27B" in body
        assert get(port, "/ui/chat/chat.js")[0] == 200 and get(port, "/ui/shell/fonts/ChakraPetch-700.woff2")[0] == 200
    else:
        assert json.loads(body) == {"status": "ok"}
        assert get(port, "/ui/chat/chat.js")[0] == 404
    assert json.loads(get(port, "/health")[2]) == {"status": "ok"}
    assert json.loads(get(port, "/v1/models")[2])["data"][0]["id"] == "qwen3-8-27b"


def test_cli_has_the_ui_flag(capsys):
    from mashbox import cli
    with pytest.raises(SystemExit) as e:
        cli.main(["run", "qwen3-8-27b", "--help"])
    assert e.value.code == 0 and "--ui" in capsys.readouterr().out


def test_standalone_page_streams_from_any_server():
    sp, pp = free_port(), free_port()
    httpd, log = chat_stub.serve(sp, model="stub-model")
    from mashbox.ui.chat import __main__ as M
    threading.Thread(target=M.main, args=(["--api", f"http://127.0.0.1:{sp}", "--port", str(pp)],), daemon=True).start()
    for _ in range(50):
        try:
            code, typ, body = get(pp, "/")
            break
        except OSError:
            time.sleep(0.05)
    assert code == 200 and blocks(body.decode())[0]["name"] == "stub-model"
    assert json.loads(get(pp, "/v1/models")[2])["data"][0]["max_model_len"] == 4096
    c = http.client.HTTPConnection("127.0.0.1", pp, timeout=10)
    c.request("POST", "/v1/chat/completions", body=json.dumps({"model": "stub-model", "stream": True, "stream_options": {"include_usage": True},
              "messages": [{"role": "user", "content": "hi"}], "temperature": 0.3, "chat_template_kwargs": {"enable_thinking": True}}),
              headers={"Content-Type": "application/json"})
    r = c.getresponse()
    assert r.status == 200 and r.getheader("Content-Type") == "text/event-stream"
    t, got = [], []
    while True:
        line = r.readline()
        if not line:
            break
        if line.startswith(b"data: "):
            t.append(time.monotonic())
            got.append(line[6:].strip())
    assert got[-1] == b"[DONE]" and json.loads(got[-2])["usage"]["completion_tokens"] > 10
    assert t[-1] - t[1] > 0.1, "the reply arrives in pieces, not in one buffered block"
    assert any(b"reasoning_content" in g for g in got)
    assert log["requests"][-1]["temperature"] == 0.3
    httpd.shutdown()
