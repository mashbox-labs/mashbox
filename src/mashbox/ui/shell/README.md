# MASHBOX module shell

Every MASHBOX module page (the chat, the world model, every module after them) loads these files and fills only its own
content area. The top bar, the model bar, the footer, the tokens, the fonts and the building blocks live here and
nowhere else, so changing this folder changes every module at once. Every module shares one look.

## What every module page shows

- **Top bar:** the MASHBOX mark and `MASHBOX <MODULE>`, the "Made by MASHBOX Labs" label, links back to mashbox.gg
  (Models, Docs, Research), GitHub and Hugging Face, and the light/dark switch.
- **Model bar** (stays in view on desktop): the model's name and who made it (linked to the original model), one line on
  what it does, the chips it runs on, a status light the module drives, and buttons to its docs page, its model card and
  its code.
- **Footer:** an optional Resources list, "Made by MASHBOX Labs", the company line and the links back.

## Page contract

```html
<head>
  <link rel="icon" href="ui/shell/favicon.svg" type="image/svg+xml">
  <link rel="stylesheet" href="ui/shell/shell.css">
  <script type="application/json" id="mbx-module">{"model": "qwen3-8-27b", "module": "Chat"}</script>
  <script src="ui/shell/shell.js"></script>          <!-- after the block, no defer: the theme is set before paint -->
  <style>/* the module's own styles, using the --mbx-* tokens */</style>
</head>
<body class="mbx-ui">
  <header data-mbx-top></header>                       <!-- the shell draws the top bar and the model bar -->
  <nav class="mbx-tabs mbx-wrap" data-mbx-tabs>...</nav> <!-- optional: the module's screens -->
  <main class="mbx-main">...the module...</main>
  <footer data-mbx-foot></footer>
</body>
```

Start from `template.html`. `example.html` is a complete small module (a chat screen with the shared sampler settings).

Rules (the CPU test `tests/cpu/test_module_shell.py` enforces each one):

- Link `ui/shell/shell.css` and load `ui/shell/shell.js` plainly in `<head>`, after the config block.
- `<body class="mbx-ui">`, one `<header data-mbx-top>`, one `<footer data-mbx-foot>`.
- Never copy a shell part into a page: no `--mbx-*` token definitions (except the layout settings `--mbx-ar`,
  `--mbx-reserve`, `--mbx-cols` and the accent `--mbx-acc`), no `@font-face`, no hand-written "Made by MASHBOX Labs",
  no company line, no `mbx-top` / `mbx-bar` / `mbx-foot` markup.
- Nothing from the internet at run time: no font, CSS or script CDNs.
- Every link the page shows resolves to a real docs page, model repo, code path or mashbox.gg page.

## Config API

The page's block names the model and only what differs from `models.json`. `render(page_bytes)` (the module's server
calls it on every page request) fills in the rest:

| Key | Set by | Default |
|---|---|---|
| `model` | page | the `models.json` id |
| `module` | page | from the model's `kind`: `world` = "World model", `llm`/`llm-tpu` = "Chat" |
| `name`, `maker`, `line` | `models.json` | `name`, `upstream.maker`, `summary` |
| `chips` | `models.json` | `TPU <chip>` for each chip whose status is `passing`; a page may narrow it (`["TPU v6e-4"]`) |
| `links` | `models.json` | `docs` mashbox.gg/docs/models/<id>, `card` the HF repo, `code` the folder at the release tag, `upstream` the original model; a page may add `extra: [{label, url}]` |
| `theme` | page | `system` (or `dark`, `light`); the viewer's switch wins and is remembered |
| `status` | page | none; `{state: live / busy / error / off, text}` for the first paint |
| `resources` | page | none; `[{title, by, url}]`, listed above the footer |
| `sticky` | page | `true`: the model bar stays in view on desktop |
| `foot_links` | page | none; extra `[{label, url}]` in the footer |
| `site`, `company` | shell only | the links back and the company line; a page cannot set them |

Python (`mashbox.ui.shell`): `render(page)`, `asset(path)` (bytes and type for any `ui/shell/...` request, nothing
else), `config(model_id, **page_keys)`, `links(id)`, `chips(id)`, `site()`, `check_page(html)`,
`check_links(cfg, repo_root)`, `module_pages(repo_root)`, `ROUTE`, `COMPANY_LINE`.

A module server serves the shell like this:

```python
from mashbox.ui import shell
page = shell.render(page_bytes)                  # GET /
hit = shell.asset(path)                          # GET /ui/shell/...  -> (bytes, content type) or None
```

Browser (`window.MBX`): `MBX.config`, `MBX.status('live' | 'busy' | 'error' | 'off', text)`, `MBX.theme('light' |
'dark' | 'system')`, and the `mbx:ready` event after the shell has drawn.

Command line:

```
python -m mashbox.ui.shell serve PAGE [--port 8800]   # http://127.0.0.1:8800/
python -m mashbox.ui.shell build PAGE OUTDIR          # static copy for review
python -m mashbox.ui.shell check [PAGE ...]           # the rules above, every module page by default
```

## Tokens

Dark is the default; light swaps the same names. Use them; never redefine them.

| Token | Dark | Light | Use |
|---|---|---|---|
| `--mbx-bg` | #0b0908 | #fbf3e8 | page |
| `--mbx-panel`, `--mbx-panel-2` | #141110, #1c1714 | #fffaf3, #f6ebdc | panels, raised controls |
| `--mbx-field` | #0e0b0a | #fffdf9 | inputs |
| `--mbx-line`, `-2`, `-3` | orange at 16 / 30 / 60 % | brown at 18 / 32 %, orange 70 % | borders, hover, focus |
| `--mbx-ink`, `--mbx-dim`, `--mbx-faint` | #f3ebe1, #c9b9a7, #8f8070 | #1d140e, #5e4c3c, #8f7b68 | text |
| `--mbx-up` | #ff5a1f | #e5480f | the MASHBOX orange: primary buttons, names, live |
| `--mbx-ok`, `--mbx-down` | #ffb347, #6d84b4 | #c77800, #4f67a0 | second accent, errors |
| `--mbx-hard` | #3a1a08 | #e4c8aa | the hard offset shadow |
| `--mbx-brick`, `--mbx-mortar`, `--mbx-plate` | | | the screen band behind a plate |
| `--mbx-acc` | `--mbx-up` | | a screen's accent (set it on a tab and its section) |
| `--mbx-font`, `--mbx-mono`, `--mbx-pixel` | Chakra Petch, Share Tech Mono, Lab3x5 | | text, numbers, tiny status labels |

Type: body 15.5px (14px on phones); headings, tabs and the brand in Chakra Petch 700 uppercase (17 / 13 / 12.5px);
numbers in Share Tech Mono; the 3x5 pixel face only for short status labels. Frames are square with 2px borders and a
4px hard shadow; buttons have a 3px shadow and press in.

Blocks: `mbx-panel` (h3 inside is the panel title), `mbx-room` + `mbx-plate` + `mbx-room-line` (a screen's band),
`mbx-tabs` + `mbx-tab`, `mbx-how` (How this works), `mbx-cols`, `mbx-panels` (`--mbx-cols`), `mbx-fit` + `mbx-stage`
(`--mbx-ar`, `--mbx-reserve`), `mbx-badge`, `mbx-chip`, `mbx-seg` (the chosen button carries `.on` or `.sel`), `mbx-field`, `mbx-check`, `mbx-label`, `mbx-note`,
`mbx-pxl`, and plain `button` (`.pri`, `.sel`, `.sm`, `.ghost`), inputs, selects, textareas, checkboxes and range
sliders inside `.mbx-ui`.

Phones: under 900px columns stack and the site links get their own row; under 700px the model bar scrolls with the
page; under 480px everything packs tighter; on touch screens every field is 16px so iOS never zooms.

## Fonts

Bundled in `fonts/`: Chakra Petch 400 to 700 and Share Tech Mono 400 (Latin, SIL Open Font License 1.1, licence texts
next to them), and Lab3x5, the MASHBOX 3x5 pixel face.
