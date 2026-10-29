/* MASHBOX module shell: sets the light or dark theme before the page paints, then draws the top bar ("Made by MASHBOX
   Labs", links back), the model bar (model, maker, what it does, chips, status, docs) and the footer from the page's
   config block, and runs the module's tabs. It makes no network requests.
   Load it in <head>, after the config block and without defer. Contract: README.md next to this file.
   Public API: window.MBX = { config, status(state, text), theme(name) }. */
(function () {
  'use strict';
  var KEY = 'mbx-theme';
  var root = document.documentElement;
  var block = document.getElementById('mbx-module');
  var cfg = {};
  try { cfg = block ? JSON.parse(block.textContent || '{}') : {}; } catch (e) { cfg = {}; }

  // ---------- theme: saved choice, else the page's default, else the system setting ----------
  var media = window.matchMedia ? window.matchMedia('(prefers-color-scheme: light)') : null;
  function saved() { try { return localStorage.getItem(KEY); } catch (e) { return null; } }
  function resolve(t) { return t === 'light' || t === 'dark' ? t : (media && media.matches ? 'light' : 'dark'); }
  var choice = saved() || cfg.theme || 'system';
  root.dataset.theme = resolve(choice);
  if (media && media.addEventListener) media.addEventListener('change', function () { if (choice === 'system') root.dataset.theme = resolve('system'); });
  function setTheme(t) {
    choice = t || (root.dataset.theme === 'dark' ? 'light' : 'dark');
    root.dataset.theme = resolve(choice);
    try { localStorage.setItem(KEY, choice); } catch (e) {}
  }

  // ---------- small DOM helpers (config text always goes in as text, never as markup) ----------
  var SVG = 'http://www.w3.org/2000/svg';
  function el(tag, attrs, kids) {
    var n = document.createElement(tag);
    for (var k in attrs || {}) if (attrs[k] != null) n.setAttribute(k, attrs[k]);
    (kids || []).forEach(function (c) { if (c != null) n.appendChild(typeof c === 'string' ? document.createTextNode(c) : c); });
    return n;
  }
  function link(href, text, cls, title) {
    var a = el('a', { href: href, class: cls || null, title: title || null }, [text]);
    if (/^https?:/.test(href) && href.indexOf(location.host) < 0) { a.target = '_blank'; a.rel = 'noopener'; }
    return a;
  }
  function svg(viewBox, paths) {
    var s = document.createElementNS(SVG, 'svg');
    s.setAttribute('viewBox', viewBox); s.setAttribute('aria-hidden', 'true');
    paths.forEach(function (p) {
      var n = document.createElementNS(SVG, p[0]);
      for (var k in p[1]) n.setAttribute(k, p[1][k]);
      s.appendChild(n);
    });
    return s;
  }
  // the MASHBOX mark: a box split by the orange slash
  var uid = 0;
  function mark() {
    var box = 'M7 7H25V25H7Z M11.5 11.5V20.5H20.5V11.5Z', n = ++uid;
    var s = svg('4 4 24 24', [
      ['clipPath', { id: 'mbx-cl' + n }], ['clipPath', { id: 'mbx-cr' + n }],
      ['path', { d: box, 'fill-rule': 'evenodd', fill: 'currentColor', transform: 'translate(-0.53 1.51)', 'clip-path': 'url(#mbx-cl' + n + ')' }],
      ['path', { d: box, 'fill-rule': 'evenodd', fill: 'currentColor', transform: 'translate(0.53 -1.51)', 'clip-path': 'url(#mbx-cr' + n + ')' }],
      ['path', { d: 'M18.29 4.44H21.89L13.71 27.56H10.11Z', style: 'fill: var(--mbx-up)' }]
    ]);
    var cps = s.querySelectorAll('clipPath');
    var l = document.createElementNS(SVG, 'path'); l.setAttribute('d', 'M-32 -3.2H20.99L7.41 35.2H-32Z'); cps[0].appendChild(l);
    var r = document.createElementNS(SVG, 'path'); r.setAttribute('d', 'M24.59 -3.2H64V35.2H11.01Z'); cps[1].appendChild(r);
    return s;
  }
  var site = cfg.site || [];   // [{label, url}], from the package (mashbox.ui.shell.SITE)

  // ---------- top bar ----------
  function top(host) {
    host.classList.add('mbx-top', 'mbx-wrap');
    var home = (site[0] && site[0].url) || null;
    var brandTitle = el('b', null, ['MASHBOX ']);
    if (cfg.module) brandTitle.appendChild(el('em', null, [cfg.module]));
    var brand = el('a', { class: 'mbx-brand', href: home, 'aria-label': 'MASHBOX Labs home' }, [mark(), brandTitle]);
    var made = link(home || '#', 'Made by MASHBOX Labs', 'mbx-made');
    var nav = el('nav', { class: 'mbx-site', 'aria-label': 'MASHBOX Labs' });
    site.slice(1).forEach(function (s) { nav.appendChild(link(s.url, s.label)); });
    var btn = el('button', { class: 'mbx-icon-btn', type: 'button', title: 'Switch light or dark', 'aria-label': 'Switch light or dark' }, [
      svg('0 0 16 16', [['circle', { cx: 8, cy: 8, r: 5.5, fill: 'none', stroke: 'currentColor', 'stroke-width': 1.4 }],
        ['path', { d: 'M8 2.5a5.5 5.5 0 0 1 0 11z', fill: 'currentColor' }]])]);
    btn.addEventListener('click', function () { setTheme(); });
    host.appendChild(el('div', { class: 'mbx-top-in' }, [brand, made, nav, btn]));

    // model bar
    if (!cfg.name) return;
    var l = cfg.links || {};
    var who = el('span', { class: 'mbx-bar-who' }, [el('b', { class: 'mbx-bar-name' }, [cfg.name])]);
    if (cfg.maker) who.appendChild(el('span', { class: 'mbx-bar-by' }, ['by ', l.upstream ? link(l.upstream, cfg.maker, null, 'The original model') : cfg.maker]));
    var end = el('span', { class: 'mbx-bar-end' });
    (cfg.chips || []).forEach(function (c) { end.appendChild(el('span', { class: 'mbx-badge mbx-hw' }, [c])); });
    var live = el('span', { class: 'mbx-live', role: 'status', hidden: '' }, [el('i'), el('span')]);
    end.appendChild(live);
    [['docs', 'Docs', 'How to run it, settings and results'], ['card', 'Model card', 'The model card on Hugging Face'], ['code', 'Code', 'The code on GitHub']]
      .forEach(function (k) { if (l[k[0]]) end.appendChild(link(l[k[0]], k[1], 'mbx-btn sm', k[2])); });
    (l.extra || []).forEach(function (x) { end.appendChild(link(x.url, x.label, 'mbx-btn sm ghost')); });
    var bar = el('section', { class: 'mbx-bar', 'aria-label': 'Model' }, [el('span', { class: 'mbx-bar-lab' }, ['Model']), who,
      cfg.line ? el('span', { class: 'mbx-bar-line' }, [cfg.line]) : null, end]);
    var wrap = el('div', { class: 'mbx-bar-wrap' + (cfg.sticky === false ? ' mbx-static' : '') }, [el('div', { class: 'mbx-wrap' }, [bar])]);
    host.parentNode.insertBefore(wrap, host.nextSibling);
    MBX._live = live;
    if (cfg.status) status(cfg.status.state, cfg.status.text);
  }

  // ---------- footer ----------
  function foot(host) {
    host.classList.add('mbx-foot', 'mbx-wrap');
    var res = cfg.resources || [];
    if (res.length) {
      var ol = el('ol', { class: 'mbx-res' });
      res.forEach(function (r) { ol.appendChild(el('li', null, [link(r.url, r.title), r.by ? ', ' + r.by : null])); });
      host.appendChild(el('section', { class: 'mbx-res-wrap', 'aria-label': 'Resources' }, [el('h2', null, ['Resources']), ol]));
    }
    var links = el('nav', { class: 'mbx-foot-links', 'aria-label': 'More from MASHBOX Labs' });
    site.slice(1).concat(cfg.foot_links || []).forEach(function (s) { links.appendChild(link(s.url, s.label)); });
    if (site[0]) links.appendChild(link(site[0].url, site[0].url.replace(/^https?:\/\//, '').replace(/\/$/, '')));
    host.appendChild(el('div', { class: 'mbx-foot-in' }, [
      el('a', { class: 'mbx-foot-brand', href: (site[0] && site[0].url) || null }, [mark(), 'Made by MASHBOX Labs']),
      cfg.company ? el('p', { class: 'mbx-foot-line' }, [cfg.company]) : null, links]));
  }

  // ---------- tabs: <nav class="mbx-tabs" data-mbx-tabs><button data-tab="x"> shows <section class="mbx-tab" id="tab-x"> ----------
  function tabs(nav) {
    var btns = nav.querySelectorAll('button[data-tab]');
    function show(name) {
      btns.forEach(function (b) { var on = b.dataset.tab === name; b.classList.toggle('on', on); b.setAttribute('aria-selected', on); });
      document.querySelectorAll('.mbx-tab').forEach(function (t) { t.classList.toggle('on', t.id === 'tab-' + name); });
    }
    nav.setAttribute('role', 'tablist');
    btns.forEach(function (b) { b.setAttribute('role', 'tab'); b.addEventListener('click', function () { show(b.dataset.tab); }); });
    var first = nav.querySelector('button.on') || btns[0];
    if (first) show(first.dataset.tab);
  }

  // ---------- status: MBX.status('live' | 'busy' | 'error' | 'off', text) ----------
  function status(state, text) {
    var live = MBX._live;
    if (!live) return;
    live.hidden = !state;
    live.dataset.state = state || '';
    live.lastChild.textContent = text || ({ live: 'Live', busy: 'Working', error: 'Down', off: 'Off' })[state] || '';
  }

  var MBX = window.MBX = { config: cfg, status: status, theme: setTheme, _live: null };
  function draw() {
    document.querySelectorAll('[data-mbx-top]').forEach(top);
    document.querySelectorAll('[data-mbx-foot]').forEach(foot);
    document.querySelectorAll('[data-mbx-tabs]').forEach(tabs);
    document.dispatchEvent(new CustomEvent('mbx:ready', { detail: cfg }));
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', draw); else draw();
})();
