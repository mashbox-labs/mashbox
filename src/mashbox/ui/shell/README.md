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

Dark is the default; light swaps the same names (`data-theme="light"` on the root, or a light system setting when the
page sets no theme). Use them; never redefine them.

| Token | Dark | Light | Use |
|---|---|---|---|
| `--mbx-bg` | #0b0a09 | #f6f3ed | page |
| `--mbx-panel`, `--mbx-panel-2` | #171411, #201c19 | #ece7dd, #e6e0d4 | tiles and bars, the segmented control's track |
| `--mbx-sel` | #211d19 | #e2dccf | the picked tab, hover fills; `--mbx-tag` (tags) is `--mbx-sel` |
| `--mbx-field` | #0b0a09 | #f6f3ed | inputs |
| `--mbx-line`, `-2`, `-3` | #24201c, #37312b, #ff5a1f | #e0dacd, #cdc5b6, #ff5a1f | strokes: rule, field and badge, ember |
| `--mbx-ink`, `--mbx-dim`, `--mbx-faint` | #ece6da, #b3ab9e, #857d71 | #0b0a09, #4d4740, #776f65 | text |
| `--mbx-up` | #ff5a1f | #ff5a1f | ember, the one accent: the live light, the "Made by" stroke, focus rings |
| `--mbx-accent` | #ff8a5c | #b8400f | ember text and links |
| `--mbx-ok`, `--mbx-down`, `--mbx-good` | #fbbf24, #f87171, #4ade80 | #a16207, #b91c1c, #15803d | busy, error, good |
| `--mbx-btn`, `--mbx-on-btn` | #ece6da, #0b0a09 | #0b0a09, #f6f3ed | the primary pill button and its text |
| `--mbx-btn-2`, `--mbx-btn-in` | #201c19, #211d19 | #e6e0d4, #e2dccf | other buttons; a button on a tile |
| `--mbx-shadow` | soft drop shadow | the same | the top bar |
| `--mbx-hard` | transparent | transparent | no hard shadows: a shadow drawn in it is invisible |
| `--mbx-brick`, `--mbx-mortar`, `--mbx-plate` | #171411, #171411, #0b0a09 | #ece7dd, #ece7dd, #f6f3ed | the screen band behind a plate |
| `--mbx-stage` | #000000 | #000000 | behind video and canvas |
| `--mbx-acc` | `--mbx-up` | | a screen's accent (set it on a tab and its section) |
| `--mbx-font`, `--mbx-mono` | Outfit, Geist Mono ASCII | | text; numbers and code |
| `--mbx-pixel`, `--mbx-disp` | `--mbx-mono`, Big Shoulders Display | | tiny labels; a display title (no shell part uses it) |
| `--mbx-r-pill`, `--mbx-r-lg`, `--mbx-r`, `--mbx-r-sm` | 999px, 18px, 14px, 8px | | pills; tiles and bars; small tiles; fields |

Hover fills (`--mbx-btn-h`, `--mbx-btn-2h`, `--mbx-btn-inh`), `--mbx-on-up` (text on ember), `--mbx-soft` (a
translucent ember) and `--mbx-cold` (the same as `--mbx-faint`) complete the colours. Layout: `--mbx-max`
1440px, `--mbx-gutter` 20px, `--mbx-gap` 14px.

Type: Outfit for text, with no uppercase: body 400 15.5px (14.5px under 480px wide), panel headings 700 16.5px, the model
name in the model bar 700 20px, tabs, site links and buttons 500 15px (the picked tab 600), the brand 650 15px. Numbers
and code in Geist Mono ASCII (`mbx-mono`, tabular figures); `mbx-pxl` is a 13px Geist Mono label in the screen's
accent. The pixel mark is the one 8-bit touch.

Shapes: an element has a full, thin (1px), rounded stroke or no stroke at all, never one side. Buttons, tags and the top
bar are pills. Buttons have no stroke and no shadow (the primary in `--mbx-btn`, the others a warm fill) and move down
1px when pressed. Panels, cards, the model bar and the footer are filled rounded tiles (18px) with no stroke. Fields
have a 1px `--mbx-line-2` stroke at radius 8 and a 2px ember focus ring; a badge is a 1px `--mbx-line-2` pill with no
fill; the "Made by MASHBOX Labs" label and a room's plate are 1px ember pills. The top bar carries the soft
`--mbx-shadow`; nothing has a hard offset shadow.

Blocks: `mbx-panel` and `mbx-card` (an h2 or h3 inside is the title), `mbx-room` + `mbx-plate` + `mbx-room-line` (a
screen's band), `mbx-tabs` + `mbx-tab`, `mbx-how` (How this works), `mbx-cols`, `mbx-panels` (`--mbx-cols`), `mbx-fit` +
`mbx-stage` (`--mbx-ar`, `--mbx-reserve`), `mbx-badge`, `mbx-chip`, `mbx-seg` (the chosen button carries `.on` or `.sel`),
`mbx-field`, `mbx-check`, `mbx-label`, `mbx-note`, `mbx-pxl`, and plain `button` or `.mbx-btn` (`.pri` or `.primary`,
`.sel`, `.sm`, `.ghost`), inputs, selects, textareas, checkboxes and range sliders inside `.mbx-ui`.

Phones: under 1100px panels take at most two columns; under 900px columns stack and the site links get their own row;
under 700px the model bar scrolls with the page; under 480px everything packs tighter; on touch screens every field is
16px so iOS never zooms.

## Fonts

`shell.css` loads three faces from `fonts/`, each under the SIL Open Font License, Version 1.1, with its licence text
next to it: Outfit (variable, weights 100 to 900; `OFL-Outfit.txt`), Geist Mono ASCII (`OFL-Geist.txt`) and Big
Shoulders Display 900 (`OFL-BigShoulders.txt`).

The folder also holds Chakra Petch 400 to 700 and Share Tech Mono 400 (SIL Open Font License, Version 1.1;
`OFL-ChakraPetch.txt`, `OFL-ShareTechMono.txt`) and Lab3x5, the MASHBOX 3x5 pixel face. The shell's stylesheet does not
load them.
