"""Drive the MASHBOX Chat page in a real browser (headless Chromium, its own context) and record what happened.

    python tests/cpu/chat_ui_e2e.py --url http://127.0.0.1:8801/ --shots DIR [--stub http://127.0.0.1:8911] [--timeout 180]

With --stub (tests/cpu/chat_stub.py behind the page) it also checks every setting reached the server. Without it the
page talks to a real model server; the replies and timings are the model's. Prints one JSON object: checks (name ->
true/false/detail), replies (the measured line under each reply), screenshots. Exit 1 when a check fails.
"""
import argparse
import json
import pathlib
import sys
import time
import urllib.request
from urllib.parse import urlsplit

from playwright.sync_api import sync_playwright

ap = argparse.ArgumentParser()
ap.add_argument("--url", required=True)
ap.add_argument("--shots", required=True)
ap.add_argument("--stub")
ap.add_argument("--timeout", type=float, default=60)
a = ap.parse_args()
shots = pathlib.Path(a.shots)
shots.mkdir(parents=True, exist_ok=True)
T = int(a.timeout * 1000)
checks, replies, files, errors, foreign = {}, [], [], [], set()
host = urlsplit(a.url).netloc


def ok(name, cond, detail=None):
    checks[name] = bool(cond) if detail is None else (detail if cond else f"FAIL: {detail}")


def stub_last():
    return json.loads(urllib.request.urlopen(a.stub + "/stub/requests", timeout=5).read())


def done(page, n):
    page.wait_for_function(f"() => !document.querySelector('.msg.bot.live') && document.querySelectorAll('.msg.bot').length >= {n}", timeout=T)
    s = page.locator(".msg.bot").last.locator(".stats").inner_text()
    replies.append(s)
    return s


def shot(page, name, full=False):
    p = shots / f"{name}.png"
    page.screenshot(path=str(p), full_page=full)
    files.append(str(p))


def no_sideways(page):
    return page.evaluate("() => document.documentElement.scrollWidth <= window.innerWidth + 1")


def send(page, text):
    page.fill("#msg", text)
    page.press("#msg", "Enter")


with sync_playwright() as pw:
    br = pw.chromium.launch()
    ctx = br.new_context(viewport={"width": 1440, "height": 900}, accept_downloads=True, color_scheme="dark")
    ctx.grant_permissions(["clipboard-read", "clipboard-write"])
    page = ctx.new_page()
    page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.on("request", lambda r: foreign.add(urlsplit(r.url).netloc) if urlsplit(r.url).netloc not in (host, "") else None)
    t0 = time.monotonic()
    page.goto(a.url)
    page.wait_for_function("() => (document.querySelector('.mbx-live') || {}).textContent === 'Ready'", timeout=T)
    ok("page ready (s)", True, round(time.monotonic() - t0, 2))
    ok("shell top bar says Made by MASHBOX Labs", page.locator(".mbx-made").inner_text().lower() == "made by mashbox labs")
    ok("model bar names the model", page.locator(".mbx-bar-name").inner_text(), page.locator(".mbx-bar-name").inner_text())
    ok("footer carries the company line", "MASHBOX Labs makes open models run faster" in page.locator(".mbx-foot-line").inner_text())
    ctx_txt = page.locator("#ctx-t").inner_text()
    ok("context size from the server", "/ ?" not in ctx_txt, ctx_txt)
    shot(page, "chat-1440-dark-empty")

    rec = page.locator('#preset [data-p="rec"]')
    if rec.is_visible():
        rec.click()
        ok("recommended preset fills the model maker's values", page.input_value("#k-temperature") != "", page.input_value("#k-temperature"))
    page.click('#preset [data-p="greedy"]')
    ok("greedy preset sets temperature 0", page.input_value("#k-temperature") == "0")
    if a.stub:
        page.fill("#k-temperature", "0.6"); page.press("#k-temperature", "Tab")
        page.fill("#k-top_k", "15"); page.press("#k-top_k", "Tab")
        page.fill("#k-presence_penalty", "0.5"); page.press("#k-presence_penalty", "Tab")
        page.fill("#k-max", "300"); page.press("#k-max", "Tab")
        page.fill("#k-seed", "7"); page.press("#k-seed", "Tab")
        page.fill("#k-stop", "STOPHERE")
        page.fill("#system", "Be brief.")

    # 1. a reply, streamed, with its measured line
    send(page, "Hello! In one short sentence, what is a TPU?")
    page.wait_for_selector(".msg.bot.live", timeout=T)
    ok("reply streams (live state seen)", True)
    s1 = done(page, 1)
    ok("reply shows first token time and speed", "first token" in s1 and "tok/s" in s1, s1)
    if a.stub:
        b = stub_last()["requests"][-1]
        want = {"temperature": 0.6, "top_k": 15, "presence_penalty": 0.5, "max_tokens": 300, "seed": 7, "stop": ["STOPHERE"]}
        ok("every setting reached the server", all(b.get(k) == v for k, v in want.items()) and b["messages"][0] == {"role": "system", "content": "Be brief."}
           and b["stream"] and b["stream_options"] == {"include_usage": True}, json.dumps({k: b.get(k) for k in want}))
    ok("context meter counts the reply", not page.locator("#ctx-t").inner_text().startswith("0 /"), page.locator("#ctx-t").inner_text())

    # 2. regenerate keeps both replies
    page.locator(".msg.bot").last.get_by_role("button", name="Regenerate").click()
    done(page, 1)
    ok("regenerate keeps both versions", page.locator(".msg.bot .vers").inner_text().replace("\n", " ").strip().endswith("2 / 2 ›"), page.locator(".msg.bot .vers").inner_text())
    page.locator(".msg.bot .vers button").first.click()
    ok("switch back to the first version", "1 / 2" in page.locator(".msg.bot .vers").inner_text())

    # 3. multi-turn, then edit the first message (replaces what follows)
    send(page, "Now say the same thing for a GPU, in a Markdown table with two rows.")
    done(page, 2)
    if a.stub:
        ok("the second turn carries the history", len(stub_last()["requests"][-1]["messages"]) == 4)
    page.locator(".msg.user").first.get_by_role("button", name="Edit").click()
    page.fill(".msg.editing textarea", "Hello! In five words, what is a TPU?")
    page.locator(".msg.editing").get_by_role("button", name="Send").click()
    done(page, 1)
    ok("edit resends and drops what followed", page.locator(".msg").count() == 2, page.locator(".msg").count())

    # 4. Markdown code block with a copy button
    send(page, "Write a Python one-liner that prints hello. Put it in a fenced code block and nothing else.")
    done(page, 2)
    cb = page.locator(".msg.bot").last.locator(".cb")
    ok("code block rendered", cb.count() >= 1, cb.count())
    if cb.count():
        cb.first.locator("[data-copy]").click()
        ok("code copy button answers", page.wait_for_function("(b) => b.textContent === 'Copied'", arg=cb.first.locator("[data-copy]").element_handle(), timeout=3000))

    # 5. thinking, collapsible
    if page.locator("#think").is_visible():
        page.click('#think [data-think="1"]')
        send(page, "What is 17 times 23? Answer with the number.")
        page.wait_for_selector(".msg.bot.live details.think", timeout=T)
        ok("thinking streams into an open block", page.locator(".msg.bot.live details.think").get_attribute("open") is not None)
        shot(page, "chat-1440-dark-thinking")
        done(page, 3)
        th = page.locator(".msg.bot").last.locator("details.think")
        ok("thinking folds away when the answer comes", th.count() == 1 and th.get_attribute("open") is None)
        th.locator("summary").click()
        ok("thinking opens on click", len(th.locator(".think-b").inner_text()) > 0, len(th.locator(".think-b").inner_text()))
        page.click('#think [data-think="0"]')

    # 6. stop
    n_bot = page.locator(".msg.bot").count()
    send(page, "Write a long story about a lighthouse in a storm, at least 800 words.")
    page.wait_for_function("() => { const m = document.querySelector('.msg.bot.live .md'); return m && m.textContent.length > 40; }", timeout=T)
    page.keyboard.press("Escape")
    s = done(page, n_bot + 1)
    ok("Esc stops the reply and keeps the text", "stopped" in s and len(page.locator(".msg.bot").last.locator(".md").inner_text()) > 40, s)
    t_stop = time.monotonic()
    send(page, "Say OK.")
    s_after = done(page, n_bot + 2)
    ok("next reply after a stop (s, send to done)", True, round(time.monotonic() - t_stop, 2))

    shot(page, "chat-1440-dark", full=False)
    page.evaluate("MBX.theme('light')")
    page.wait_for_timeout(400)
    shot(page, "chat-1440-light")
    ok("desktop: no sideways scroll", no_sideways(page))

    # 7. history, new chat, reload
    page.keyboard.press("Control+Shift+O")
    ok("new chat shortcut empties the log", page.locator("#log .empty").count() == 1)
    send(page, "Hi")
    done(page, 1)
    page.reload()
    page.wait_for_function("() => (document.querySelector('.mbx-live') || {}).textContent === 'Ready'", timeout=T)
    items = page.locator("#hist li .h-open")
    ok("chats survive a reload", items.count() >= 2, items.count())
    items.nth(1).click()
    ok("opening an older chat shows it", page.locator(".msg").count() >= 4, page.locator(".msg").count())
    with page.expect_download() as dl:
        page.click("#export-md")
    p = shots / "export.md"
    dl.value.save_as(str(p))
    ok("export as Markdown", "## You" in p.read_text())
    with page.expect_download() as dl:
        page.click("#export-json")
    j = json.loads(dl.value.path().read_text())
    ok("export as JSON (OpenAI messages + settings)", j["format"] == "mashbox-chat" and j["messages"][0]["role"] in ("system", "user") and "temperature" in j["settings"])

    page.locator("body").click(position={"x": 5, "y": 5})
    page.keyboard.press("?")
    ok("? opens the shortcuts", page.locator("#keys-dlg").is_visible())
    page.click("#keys-close")
    page.click('.mbx-tabs [data-tab="api"]')
    ok("API screen builds the request", "/v1/chat/completions" in page.locator("#call").inner_text())
    shot(page, "api-1440-light")
    page.click('.mbx-tabs [data-tab="chat"]')

    # 8. phone
    ph = br.new_context(viewport={"width": 390, "height": 844}, device_scale_factor=3, is_mobile=True, has_touch=True, color_scheme="light")
    pp = ph.new_page()
    pp.on("pageerror", lambda e: errors.append("phone: " + str(e)))
    pp.goto(a.url)
    pp.wait_for_function("() => (document.querySelector('.mbx-live') || {}).textContent === 'Ready'", timeout=T)
    ok("phone: no sideways scroll", no_sideways(pp))
    pp.fill("#msg", "Hello from a phone. Answer in one sentence.")
    pp.tap("#send")
    sp = done(pp, 1)
    ok("phone: reply with measured line", "first token" in sp, sp)
    ok("phone: composer stays on screen", pp.evaluate("() => { const r = document.querySelector('#composer').getBoundingClientRect(); return r.bottom <= innerHeight + 1 && r.top < innerHeight; }"))
    shot(pp, "chat-390-light")
    pp.evaluate("window.scrollTo(0, 0)")
    shot(pp, "chat-390-light-top")
    pp.evaluate("document.querySelector('#composer').scrollIntoView({block: 'end'})")
    pp.tap("#open-set")
    pp.wait_for_timeout(300)
    ok("phone: settings drawer opens", pp.locator("#set").is_visible() and pp.evaluate("() => document.querySelector('#set').getBoundingClientRect().left >= 0"))
    shot(pp, "settings-390-light")
    pp.tap("#scrim", position={"x": 10, "y": 400})
    pp.tap("#open-chats")
    pp.wait_for_timeout(300)
    ok("phone: chat list drawer opens", pp.evaluate("() => document.querySelector('#side').getBoundingClientRect().left >= 0"))
    shot(pp, "chats-390-light")
    pp.tap("#scrim", position={"x": 380, "y": 400})
    pp.wait_for_timeout(400)
    ok("phone: drawers close on the backdrop", not pp.locator("#side").is_visible() or pp.evaluate("() => document.querySelector('#side').getBoundingClientRect().right <= 0"))
    pp.evaluate("MBX.theme('dark')")
    pp.evaluate("window.scrollTo(0, 0)")
    pp.wait_for_timeout(400)
    shot(pp, "chat-390-dark-top")
    ok("phone: no sideways scroll after use", no_sideways(pp))
    br.close()

ok("no page errors", not errors, "; ".join(errors[:5]) or "none")
ok("nothing fetched from another host", not foreign, ", ".join(sorted(foreign)) or "none")
bad = [k for k, v in checks.items() if v is False or (isinstance(v, str) and v.startswith("FAIL"))]
print(json.dumps(dict(checks=checks, replies=replies, screenshots=files, failed=bad), indent=1))
sys.exit(1 if bad else 0)
