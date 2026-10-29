/* MASHBOX Chat: a chat screen for any model behind an OpenAI-compatible server (POST /v1/chat/completions, GET
   /v1/models). The page around it is the MASHBOX module shell; this file fills only the content area.
   Model details come from the server (/v1/models: id, context) and from the page's #mbx-chat block (recommended
   sampling, thinking), which the server fills from models.json. Chats and settings stay in this browser.
   The pure helpers at the top (Markdown, the stream reader, the request body, reply timing) also load in Node for the
   CPU tests: tests/cpu/test_chat_module.py. */
(function (root, factory) {
  var api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.MBXChat = api;
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  // ------------------------------------------------------------------ Markdown (escaped first, so model text never becomes markup)
  var ESC = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' };
  function esc(s) { return String(s == null ? '' : s).replace(/[&<>"']/g, function (c) { return ESC[c]; }); }
  // links in model text: http(s), or mailto: with an address (mailto:javascript:... is not one)
  function safeUrl(u) { u = String(u || '').trim(); return /^(https?:\/\/|mailto:[^\s:\/?#@]+@[^\s:\/?#@]+(\?|$))/i.test(u) ? u : null; }
  function webUrl(u) { u = String(u || '').trim(); return /^https?:\/\//i.test(u) ? u : null; }

  function emph(h) {   // h is already escaped
    return h.replace(/\*\*(?=\S)([\s\S]*?\S)\*\*/g, '<strong>$1</strong>')
      .replace(/(^|[^\w])__(?=\S)([\s\S]*?\S)__(?!\w)/g, '$1<strong>$2</strong>')
      .replace(/(^|[^*\w])\*(?=[^\s*])([\s\S]*?[^\s*])\*(?!\*)/g, '$1<em>$2</em>')
      .replace(/(^|[^\w])_(?=[^\s_])([\s\S]*?[^\s_])_(?!\w)/g, '$1<em>$2</em>')
      .replace(/~~(?=\S)([\s\S]*?\S)~~/g, '<del>$1</del>');
  }

  function inline(text) {
    var slots = [];
    function keep(h) { slots.push(h); return '\u0000' + (slots.length - 1) + '\u0000'; }
    var s = String(text).replace(/\u0000/g, '');
    s = s.replace(/(`+)([^`]|[^`][\s\S]*?[^`])\1(?!`)/g, function (_, t, c) {
      return keep('<code>' + esc(/^ .* $/.test(c) ? c.slice(1, -1) : c) + '</code>');
    });
    s = s.replace(/\[([^\]\n]+)\]\(\s*<?([^)\s>]+)>?(?:\s+"[^"]*")?\s*\)/g, function (m, t, u) {
      var h = safeUrl(u);
      return h ? keep('<a href="' + esc(h) + '" target="_blank" rel="noopener noreferrer">' + emph(esc(t)) + '</a>') : m;
    });
    s = s.replace(/(^|[\s(])(https?:\/\/[^\s<>()]+[^\s<>().,;:!?'"\]])/g, function (m, pre, u) {
      return pre + keep('<a href="' + esc(u) + '" target="_blank" rel="noopener noreferrer">' + esc(u) + '</a>');
    });
    s = emph(esc(s)).replace(/ {2,}\n|\\\n/g, '<br>').replace(/\n/g, '<br>');
    return s.replace(/\u0000(\d+)\u0000/g, function (_, i) { return slots[+i]; });
  }

  var FENCE = /^ {0,3}(`{3,}|~{3,})\s*([^\s`]*)[^`]*$/;
  var LI = /^( *)([-*+]|\d{1,9}[.)])\s+(.*)$/;
  var HR = /^ {0,3}([-*_])(?:[ \t]*\1){2,}[ \t]*$/;
  var HEAD = /^ {0,3}(#{1,6})[ \t]+(.*?)(?:[ \t]+#+)?[ \t]*$/;
  var TSEP = /^ *\|? *:?-{1,}:? *(\| *:?-{1,}:? *)*\|? *$/;
  function indent(l) { return /^ */.exec(l)[0].length; }
  function blank(l) { return /^\s*$/.test(l); }
  function cells(l) { return l.trim().replace(/^\|/, '').replace(/\|$/, '').split(/(?<!\\)\|/).map(function (c) { return c.trim().replace(/\\\|/g, '|'); }); }
  function starts(l, next) {
    return FENCE.test(l) || HEAD.test(l) || HR.test(l) || /^ {0,3}>/.test(l) || LI.test(l) || (l.indexOf('|') >= 0 && next != null && TSEP.test(next) && next.indexOf('-') >= 0);
  }

  function codeBlock(lang, body) {
    return '<div class="cb"><div class="cb-h"><span>' + esc(lang || 'text') + '</span><button type="button" class="sm ghost" data-copy>Copy</button></div>' +
      '<pre><code' + (lang ? ' data-lang="' + esc(lang) + '"' : '') + '>' + esc(body) + '</code></pre></div>';
  }

  function md(src) {
    var lines = String(src == null ? '' : src).replace(/\r\n?/g, '\n').replace(/\t/g, '    ').split('\n');
    var out = [], i = 0, n = lines.length;
    while (i < n) {
      var l = lines[i], m;
      if (blank(l)) { i++; continue; }
      if ((m = FENCE.exec(l))) {
        var mark = m[1], body = [];
        i++;
        while (i < n && !new RegExp('^ {0,3}' + mark[0] + '{' + mark.length + ',}\\s*$').test(lines[i])) body.push(lines[i++]);
        i++;   // the closing fence (absent while a reply is still streaming)
        out.push(codeBlock(m[2], body.join('\n')));
        continue;
      }
      if ((m = HEAD.exec(l))) { var lv = m[1].length; out.push('<h' + lv + '>' + inline(m[2]) + '</h' + lv + '>'); i++; continue; }
      if (HR.test(l)) { out.push('<hr>'); i++; continue; }
      if (/^ {0,3}>/.test(l)) {
        var q = [];
        while (i < n && !blank(lines[i]) && (/^ {0,3}>/.test(lines[i]) || q.length)) {
          if (!/^ {0,3}>/.test(lines[i]) && starts(lines[i], lines[i + 1])) break;
          q.push(lines[i++].replace(/^ {0,3}> ?/, ''));
        }
        out.push('<blockquote>' + md(q.join('\n')) + '</blockquote>');
        continue;
      }
      if (l.indexOf('|') >= 0 && i + 1 < n && TSEP.test(lines[i + 1]) && lines[i + 1].indexOf('-') >= 0) {
        var hd = cells(l), al = cells(lines[i + 1]).map(function (c) { return /^:-+:$/.test(c) ? 'center' : /-:$/.test(c) ? 'right' : null; });
        var t = '<div class="tbl"><table><thead><tr>' + hd.map(function (c, k) { return '<th' + (al[k] ? ' style="text-align:' + al[k] + '"' : '') + '>' + inline(c) + '</th>'; }).join('') + '</tr></thead><tbody>';
        i += 2;
        while (i < n && !blank(lines[i]) && lines[i].indexOf('|') >= 0) {
          var row = cells(lines[i++]);
          t += '<tr>' + hd.map(function (_, k) { return '<td' + (al[k] ? ' style="text-align:' + al[k] + '"' : '') + '>' + inline(row[k] || '') + '</td>'; }).join('') + '</tr>';
        }
        out.push(t + '</tbody></table></div>');
        continue;
      }
      if ((m = LI.exec(l))) {
        var base = m[1].length, ordered = /\d/.test(m[2]), first = ordered ? parseInt(m[2], 10) : 1, items = [], cur = null, loose = false;
        while (i < n) {
          var x = lines[i], mm = LI.exec(x);
          if (mm && mm[1].length <= base + 1 && /\d/.test(mm[2]) === ordered) { cur = { lines: [mm[3]], pad: mm[1].length + mm[2].length + 1 }; items.push(cur); i++; continue; }
          if (blank(x)) {
            var nx = lines[i + 1];
            if (nx != null && !blank(nx) && (indent(nx) > base || (LI.test(nx) && indent(nx) <= base + 1 && /\d/.test(LI.exec(nx)[2]) === ordered))) {
              if (indent(nx) <= base + 1) loose = true;
              cur.lines.push(''); i++; continue;
            }
            break;
          }
          if (indent(x) > base) { cur.lines.push(x.slice(Math.min(indent(x), cur.pad))); i++; continue; }
          if (!starts(x, lines[i + 1]) && cur.lines[cur.lines.length - 1] !== '') { cur.lines.push(x.trim()); i++; continue; }
          break;
        }
        var tag = ordered ? 'ol' : 'ul';
        out.push('<' + tag + (ordered && first !== 1 ? ' start="' + first + '"' : '') + '>' + items.map(function (it) {
          var txt = it.lines.join('\n'), box = /^\[([ xX])\]\s+/.exec(txt), pre = '';
          if (box) { txt = txt.slice(box[0].length); pre = '<input type="checkbox" disabled' + (box[1] !== ' ' ? ' checked' : '') + '> '; }
          var h = md(txt);
          if (!loose) h = h.replace(/^<p>([\s\S]*?)<\/p>/, '$1');
          return '<li' + (box ? ' class="task"' : '') + '>' + pre + h + '</li>';
        }).join('') + '</' + tag + '>');
        continue;
      }
      var p = [];
      while (i < n && !blank(lines[i]) && !(p.length && starts(lines[i], lines[i + 1]))) p.push(lines[i++]);
      out.push('<p>' + inline(p.join('\n')) + '</p>');
    }
    return out.join('');
  }

  // ------------------------------------------------------------------ the stream: server-sent events, one JSON object per data line
  function sseReader() {
    var buf = '';
    return function (chunk, final) {
      buf += chunk || '';
      var evs = [], k;
      while ((k = buf.search(/\r?\n\r?\n/)) >= 0) {
        var block = buf.slice(0, k);
        buf = buf.slice(k).replace(/^\r?\n\r?\n/, '');
        var data = block.split(/\r?\n/).filter(function (x) { return x.indexOf('data:') === 0; })
          .map(function (x) { return x.slice(5).replace(/^ /, ''); }).join('\n');
        if (data) evs.push(data);
      }
      if (final && buf.trim()) {
        var rest = buf.split(/\r?\n/).filter(function (x) { return x.indexOf('data:') === 0; }).map(function (x) { return x.slice(5).replace(/^ /, ''); }).join('\n');
        if (rest) evs.push(rest);
        buf = '';
      }
      return evs;
    };
  }

  // ------------------------------------------------------------------ the request: the OpenAI chat format plus the engine's settings
  var SAMPLING = ['temperature', 'top_p', 'top_k', 'min_p', 'repetition_penalty', 'presence_penalty', 'frequency_penalty'];
  function num(v) { return v === '' || v == null || !isFinite(+v) ? null : +v; }

  function toMessages(chat, opts) {
    opts = opts || {};
    var out = [];
    if (chat.system && chat.system.trim()) out.push({ role: 'system', content: chat.system });
    (chat.messages || []).forEach(function (m) {
      if (m.role === 'user') { out.push({ role: 'user', content: m.content }); return; }
      var v = m.versions ? m.versions[m.sel || 0] : m;
      if (!v || v.error || (!v.content && !v.reasoning)) return;
      var a = { role: 'assistant', content: v.content || '' };
      if (opts.keepThinking && v.reasoning) a.reasoning_content = v.reasoning;
      out.push(a);
    });
    return out;
  }

  function buildBody(model, messages, s, caps) {
    caps = caps || {};
    var b = { model: model, messages: messages, stream: true, stream_options: { include_usage: true } };
    SAMPLING.forEach(function (k) { var v = num(s[k]); if (v != null) b[k] = v; });
    var mx = num(s.max_tokens); if (mx != null && mx >= 1) b.max_tokens = Math.round(mx);
    var seed = num(s.seed); if (seed != null) b.seed = Math.round(seed);
    var stop = String(s.stop || '').split('\n').map(function (x) { return x.replace(/\\n/g, '\n'); }).filter(function (x) { return x.length; });
    if (stop.length) b.stop = stop.slice(0, 8);
    if (caps.thinking) b.chat_template_kwargs = { enable_thinking: !!s.thinking };
    return b;
  }

  // ------------------------------------------------------------------ timing of one reply, measured in this browser
  function replyStats(t) {
    var u = t.usage || null;
    var exact = !!(u && u.completion_tokens != null);
    var tokens = exact ? u.completion_tokens : null;
    var ttft = t.tFirst != null ? (t.tFirst - t.t0) / 1000 : null;
    var gen = t.tFirst != null && t.tEnd != null ? (t.tEnd - t.tFirst) / 1000 : null;
    // decode speed: the server's own timings when it sends them (as llama-server and its web UI), else measured here
    var sv = t.timings && t.timings.predicted_per_second != null ? t.timings.predicted_per_second : null;
    return {
      ttft: ttft, tokens: tokens, total: t.tEnd != null ? (t.tEnd - t.t0) / 1000 : null,
      tps: sv != null ? sv : exact && gen > 0 && tokens > 1 ? (tokens - 1) / gen : null,
      tps_src: sv != null ? 'server' : 'browser',
      prompt: u && u.prompt_tokens != null ? u.prompt_tokens : null,
      cached: u && u.prompt_tokens_details && u.prompt_tokens_details.cached_tokens != null ? u.prompt_tokens_details.cached_tokens : null,
      finish: t.finish || null, think: t.tThinkEnd != null && t.tFirst != null ? (t.tThinkEnd - t.tFirst) / 1000 : null
    };
  }

  function chatMarkdown(chat, model) {
    var o = ['# ' + (chat.title || 'Chat'), '', '- Model: ' + (chat.model || model || ''), '- Saved: ' + new Date().toISOString(), ''];
    if (chat.system) o.push('## System', '', chat.system, '');
    (chat.messages || []).forEach(function (m) {
      if (m.role === 'user') { o.push('## You', '', m.content, ''); return; }
      var v = m.versions ? m.versions[m.sel || 0] : m;
      o.push('## ' + (chat.model || 'Assistant'), '');
      if (v.reasoning) o.push('<details><summary>Thinking</summary>', '', v.reasoning, '', '</details>', '');
      o.push(v.content || (v.error ? '(error: ' + v.error + ')' : ''), '');
      if (v.stats && v.stats.tokens != null) o.push('_' + v.stats.tokens + ' tokens' + (v.stats.tps ? ', ' + v.stats.tps.toFixed(1) + ' tok/s' : '') + (v.stats.ttft != null ? ', first token ' + v.stats.ttft.toFixed(2) + ' s' : '') + '_', '');
    });
    return o.join('\n');
  }

  var API = { esc: esc, inline: inline, md: md, safeUrl: safeUrl, webUrl: webUrl, sseReader: sseReader, toMessages: toMessages, buildBody: buildBody, replyStats: replyStats,
    chatMarkdown: chatMarkdown, SAMPLING: SAMPLING };
  if (typeof document !== 'undefined') app(API);
  return API;

  // ================================================================== the page
  function app(X) {
    var STORE = 'mbx-chat-v1', SETS = 'mbx-chat-settings-v1:';
    var $ = function (id) { return document.getElementById(id); };
    // in the MASHBOX app the page carries the app's launch token, which its /v1 and /health want; elsewhere none
    var TOKEN = (document.querySelector('meta[name="mbx-token"]') || {}).content || '';
    function hdr(h) { h = h || {}; if (TOKEN) h['X-Mashbox-Token'] = TOKEN; return h; }
    var model = { id: null, ctx: null, info: {} };
    var caps = { thinking: false, thinking_default: false, keep_thinking: false, sampling: null, source: null };
    var state = { chats: [], cur: null };
    var sets = null, busy = null, started = false, saveNote = '';
    var GENERIC = { temperature: null, top_p: null, top_k: null, min_p: null, repetition_penalty: null, presence_penalty: null, frequency_penalty: null };
    var KNOBS = [
      { k: 'temperature', label: 'Temperature', min: 0, max: 2, step: 0.05, srv: 0, note: '0 is greedy: the same message gives the same reply.' },
      { k: 'top_p', label: 'Top p', min: 0.01, max: 1, step: 0.01, srv: 1 },
      { k: 'top_k', label: 'Top k', min: 0, max: 200, step: 1, srv: 0, note: '0 keeps every token.' },
      { k: 'min_p', label: 'Min p', min: 0, max: 0.99, step: 0.01, srv: 0 },
      { k: 'repetition_penalty', label: 'Repetition penalty', min: 0.5, max: 2, step: 0.01, srv: 1, note: '1 is off.' },
      { k: 'presence_penalty', label: 'Presence penalty', min: -2, max: 2, step: 0.05, srv: 0 },
      { k: 'frequency_penalty', label: 'Frequency penalty', min: -2, max: 2, step: 0.05, srv: 0 }
    ];

    function load(k, d) { try { var v = localStorage.getItem(k); return v ? JSON.parse(v) : d; } catch (e) { return d; } }
    function save(k, v) {
      try { localStorage.setItem(k, JSON.stringify(v)); return true; } catch (e) { return false; }
    }
    function uid() { return Date.now().toString(36) + Math.random().toString(36).slice(2, 8); }
    function fmtN(n) { return n == null ? '' : Number(n).toLocaleString('en-US'); }
    function cur() { return state.chats.find(function (c) { return c.id === state.cur; }) || null; }
    function el(tag, cls, text) { var n = document.createElement(tag); if (cls) n.className = cls; if (text != null) n.textContent = text; return n; }

    // ---------------- storage: every chat in this browser, newest first; trimmed when the browser is full
    function persist() {
      state.chats.sort(function (a, b) { return b.updated - a.updated; });
      var keep = state.chats.filter(function (c) { return c.messages.length; });
      while (!save(STORE, { chats: keep, cur: state.cur }) && keep.length > 1) { keep = keep.slice(0, -1); saveNote = 'This browser is out of space: the oldest chats were not saved.'; }
    }
    function restore() {
      var d = load(STORE, null);
      if (d && Array.isArray(d.chats)) { state.chats = d.chats; state.cur = d.cur; }
    }

    // ---------------- settings, remembered per model
    function recommended() {
      var s = caps.sampling;
      if (!s) return null;
      return Object.assign({}, GENERIC, (sets && sets.thinking ? s.thinking : s.plain) || s.plain || s.thinking || {});
    }
    function applyPreset(p) {
      sets.preset = p;
      var src = p === 'rec' ? recommended() : p === 'greedy' ? Object.assign({}, GENERIC, { temperature: 0 }) : GENERIC;
      KNOBS.forEach(function (k) { sets[k.k] = src && src[k.k] != null ? src[k.k] : null; });
      showSettings(); keepSets();
    }
    function keepSets() { if (model.id) save(SETS + model.id, sets); }
    function initSettings() {
      var d = load(SETS + model.id, null);
      sets = Object.assign({ preset: caps.sampling ? 'rec' : 'server', thinking: caps.thinking_default, max_tokens: null, seed: null, stop: '', system: '' }, GENERIC, d || {});
      if (!d) applyPreset(sets.preset);
    }

    // ---------------- settings panel
    function knobRow(k) {
      var wrap = el('div', 'knob'), head = el('div', 'knob-h'), lab = el('label', 'mbx-label', k.label), box = document.createElement('input');
      box.type = 'number'; box.step = k.step; box.min = k.min; box.max = k.max; box.id = 'k-' + k.k; box.placeholder = 'server'; box.inputMode = 'decimal';
      box.setAttribute('aria-label', k.label);
      lab.htmlFor = box.id;
      var r = document.createElement('input'); r.type = 'range'; r.min = k.min; r.max = k.max; r.step = k.step; r.tabIndex = -1; r.setAttribute('aria-hidden', 'true');
      head.appendChild(lab); head.appendChild(box); wrap.appendChild(head); wrap.appendChild(r);
      if (k.note) wrap.appendChild(el('span', 'mbx-note', k.note));
      function set(v, from) {
        sets[k.k] = v; sets.preset = 'custom';
        if (from !== box) box.value = v == null ? '' : v;
        if (from !== r) r.value = v == null ? k.srv : v;
        wrap.classList.toggle('unset', v == null);
        presetMarks(); keepSets(); apiCode();
      }
      r.addEventListener('input', function () { set(+r.value, r); });
      box.addEventListener('change', function () { var v = num(box.value); if (v != null) v = Math.min(k.max, Math.max(k.min, v)); set(v, null); });
      wrap._show = function () { var v = sets[k.k]; box.value = v == null ? '' : v; r.value = v == null ? k.srv : v; wrap.classList.toggle('unset', v == null); };
      return wrap;
    }
    var knobEls = [];
    function buildSettings() {
      var host = $('knobs');
      KNOBS.forEach(function (k) { var w = knobRow(k); knobEls.push(w); host.appendChild(w); });
      $('preset').addEventListener('click', function (e) { var b = e.target.closest('button[data-p]'); if (b) applyPreset(b.dataset.p); });
      [['max_tokens', 'k-max'], ['seed', 'k-seed']].forEach(function (p) {
        $(p[1]).addEventListener('change', function () { sets[p[0]] = num($(p[1]).value); keepSets(); apiCode(); });
      });
      $('k-stop').addEventListener('input', function () { sets.stop = $('k-stop').value; keepSets(); apiCode(); });
      $('system').addEventListener('input', function () {
        sets.system = $('system').value; keepSets();
        var c = cur(); if (c) { c.system = sets.system; if (c.messages.length) { c.updated = Date.now(); persist(); } }
        apiCode();
      });
      $('reset').addEventListener('click', function () { applyPreset(caps.sampling ? 'rec' : 'server'); sets.max_tokens = null; sets.seed = null; sets.stop = ''; showSettings(); keepSets(); });
    }
    function presetMarks() {
      $('preset').querySelectorAll('button').forEach(function (b) { b.classList.toggle('on', b.dataset.p === sets.preset); });
      var rec = $('preset').querySelector('[data-p="rec"]'); rec.hidden = !caps.sampling;
      var note = sets.preset === 'rec' ? 'The model maker\'s values' + (caps.thinking ? (sets.thinking ? ' for thinking.' : ' without thinking.') : '.') :
        sets.preset === 'greedy' ? 'Temperature 0: the same message gives the same reply.' :
        sets.preset === 'server' ? 'Nothing is sent: the server uses its own defaults.' : 'Your own values, remembered for this model.';
      $('preset-note').textContent = note;
      if (webUrl(caps.source) && sets.preset === 'rec') { var a = el('a', null, 'Source'); a.href = webUrl(caps.source); a.target = '_blank'; a.rel = 'noopener'; $('preset-note').append(' ', a); }
    }
    function showSettings() {
      knobEls.forEach(function (w) { w._show(); });
      $('k-max').value = sets.max_tokens == null ? '' : sets.max_tokens;
      $('k-seed').value = sets.seed == null ? '' : sets.seed;
      $('k-stop').value = sets.stop || '';
      var c = cur(); $('system').value = c ? (c.system || '') : (sets.system || '');
      presetMarks(); thinkMarks(); apiCode();
    }
    function thinkMarks() {
      $('think').hidden = !caps.thinking;
      $('think').querySelectorAll('button').forEach(function (b) { b.classList.toggle('on', (b.dataset.think === '1') === !!sets.thinking); b.setAttribute('aria-pressed', (b.dataset.think === '1') === !!sets.thinking); });
    }

    // ---------------- history list
    function newChat(focus) {
      if (busy) return;
      var c = cur();
      if (c && !c.messages.length) { if (focus) $('msg').focus(); return; }
      c = { id: uid(), title: '', model: model.id, created: Date.now(), updated: Date.now(), system: sets ? sets.system : '', messages: [] };
      state.chats.unshift(c); state.cur = c.id;
      drawAll(); drawer(null);
      if (focus) $('msg').focus();
    }
    function drawHistory() {
      var ol = $('hist'), f = ($('find').value || '').toLowerCase();
      ol.textContent = '';
      var shown = 0;
      state.chats.forEach(function (c) {
        if (!c.messages.length && c.id !== state.cur) return;
        if (f && (c.title || '').toLowerCase().indexOf(f) < 0 && !c.messages.some(function (m) { return (m.content || '').toLowerCase().indexOf(f) >= 0; })) return;
        shown++;
        var li = el('li', c.id === state.cur ? 'on' : null), b = el('button', 'h-open');
        b.type = 'button';
        b.appendChild(el('span', 'h-t', c.title || 'New chat'));
        var meta = new Date(c.updated).toLocaleString([], { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' }) + ' · ' + c.messages.length + ' msg';
        b.appendChild(el('span', 'h-m mbx-mono', meta + (c.model && c.model !== model.id ? ' · ' + c.model : '')));
        b.addEventListener('click', function () { if (busy) return; state.cur = c.id; drawAll(); drawer(null); });
        var x = el('button', 'h-del ghost sm', '×'); x.type = 'button'; x.title = 'Delete this chat'; x.setAttribute('aria-label', 'Delete ' + (c.title || 'chat'));
        x.addEventListener('click', function () {
          if (busy && busy.chat === c.id) return;
          if (!confirm('Delete "' + (c.title || 'New chat') + '"?')) return;
          state.chats = state.chats.filter(function (z) { return z !== c; });
          if (state.cur === c.id) state.cur = state.chats[0] ? state.chats[0].id : null;
          persist(); if (!cur()) newChat(); drawAll();
        });
        li.appendChild(b); li.appendChild(x); ol.appendChild(li);
      });
      if (!shown) ol.appendChild(el('li', 'h-none', f ? 'No chat matches.' : 'No chats.'));
      $('save-note').textContent = saveNote || 'Chats are saved in this browser only.';
    }

    // ---------------- the conversation
    function drawAll() { drawHistory(); drawLog(); showSettings(); drawCtx(); }
    function drawCtx() {
      var c = cur(), used = null;
      if (c) for (var i = c.messages.length - 1; i >= 0; i--) {
        var m = c.messages[i], v = m.versions ? m.versions[m.sel || 0] : null;
        if (v && v.stats && v.stats.prompt != null) { used = v.stats.prompt + (v.stats.tokens || 0); break; }
      }
      var max = model.ctx;
      $('ctx-t').textContent = (used == null ? '0' : fmtN(used)) + ' / ' + (max ? fmtN(max) : '?') + ' tokens';
      var pct = used && max ? Math.min(100, 100 * used / max) : 0;
      $('ctx-bar').style.width = pct + '%';
      $('ctx').classList.toggle('hot', pct > 85);
      $('ctx').title = 'Context used by the last reply (prompt + reply) out of the server\'s context of ' + (max ? fmtN(max) : 'unknown') + ' tokens';
    }
    // the conversation's end (its composer), not the page's: the footer stays below until the reader scrolls to it
    function convEnd() { return document.querySelector('.conv').getBoundingClientRect().bottom + window.scrollY; }
    function nearBottom() { return window.scrollY + window.innerHeight >= convEnd() - 160; }
    function toBottom() { window.scrollTo(0, Math.max(0, convEnd() - window.innerHeight + 16)); }

    function drawLog() {
      var c = cur(), log = $('log');
      log.textContent = '';
      $('conv-title').textContent = c && c.title ? c.title : 'New chat';
      if (!c || !c.messages.length) {
        var e = el('div', 'empty');
        e.appendChild(el('span', 'mbx-pxl', model.id ? 'Ready' : 'Connecting'));
        e.appendChild(el('p', null, model.id ? 'Send a message to ' + (MBX.config.name || model.id) + '.' : 'Waiting for the server.'));
        if (model.ctx) e.appendChild(el('p', 'mbx-note', 'Context ' + fmtN(model.ctx) + ' tokens' + (caps.thinking ? ' · thinking ' + (sets.thinking ? 'on' : 'off') : '') + ' · press ? for shortcuts'));
        log.appendChild(e);
        return;
      }
      c.messages.forEach(function (m, i) { log.appendChild(m.role === 'user' ? userEl(c, m, i) : botEl(c, m, i)); });
    }
    function actions(list) {
      var row = el('div', 'acts');
      list.forEach(function (a) { if (!a) return; var b = el('button', 'sm ghost', a[0]); b.type = 'button'; b.title = a[2] || a[0]; b.addEventListener('click', a[1]); row.appendChild(b); });
      return row;
    }
    function copy(text, btn) {
      function ok() { if (btn) { var t = btn.textContent; btn.textContent = 'Copied'; setTimeout(function () { btn.textContent = t; }, 1200); } }
      if (navigator.clipboard && window.isSecureContext) navigator.clipboard.writeText(text).then(ok, function () { fallback(); });
      else fallback();
      function fallback() { var t = el('textarea'); t.value = text; t.style.position = 'fixed'; t.style.opacity = '0'; document.body.appendChild(t); t.select(); try { document.execCommand('copy'); ok(); } catch (e) {} t.remove(); }
    }
    function userEl(c, m, i) {
      var d = el('div', 'msg user'); d.dataset.i = i;
      var b = el('div', 'bubble', m.content);
      d.appendChild(b);
      d.appendChild(actions([
        ['Edit', function () { editUser(c, i, d); }, 'Edit and send again'],
        ['Copy', function (e) { copy(m.content, e.currentTarget); }]
      ]));
      return d;
    }
    function editUser(c, i, d) {
      if (busy) return;
      var m = c.messages[i], t = el('textarea', 'edit'); t.value = m.content; t.rows = Math.min(12, Math.max(2, m.content.split('\n').length));
      d.textContent = ''; d.classList.add('editing'); d.appendChild(t);
      var row = el('div', 'mbx-row edit-row');
      var cancel = el('button', 'sm ghost', 'Cancel'), go = el('button', 'sm pri', 'Send');
      cancel.type = go.type = 'button';
      cancel.addEventListener('click', function () { drawLog(); });
      go.addEventListener('click', function () {
        var v = t.value.trim(); if (!v) return;
        m.content = v; c.messages = c.messages.slice(0, i + 1); if (i === 0) c.title = title(v);
        drawLog(); reply(c);
      });
      t.addEventListener('keydown', function (e) {
        if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) { e.preventDefault(); go.click(); }
        if (e.key === 'Escape') { e.preventDefault(); drawLog(); }
      });
      row.appendChild(el('span', 'mbx-note', 'Sending again replaces everything after this message.'));
      row.appendChild(el('span', 'mbx-spacer')); row.appendChild(cancel); row.appendChild(go);
      d.appendChild(row); t.focus(); t.setSelectionRange(t.value.length, t.value.length);
    }
    function statLine(v) {
      var s = v.stats || {}, parts = [];
      if (s.ttft != null) parts.push(['first token', s.ttft.toFixed(2) + ' s', 'Time from sending to the first token, measured in this browser']);
      if (s.tps != null) parts.push(['speed', s.tps.toFixed(1) + ' tok/s', s.tps_src === 'server'
        ? 'Decode speed from the server: tokens after the first, divided by the time from the first to the last token'
        : 'Tokens after the first, divided by the time after the first token, measured in this browser']);
      if (s.tokens != null) parts.push(['reply', fmtN(s.tokens) + ' tok', 'Tokens in this reply, as the server counted them']);
      if (s.think != null) parts.push(['thinking', s.think.toFixed(1) + ' s']);
      if (s.prompt != null) parts.push(['prompt', fmtN(s.prompt) + ' tok' + (s.cached ? ' (' + fmtN(s.cached) + ' cached)' : ''), 'Prompt tokens; cached = reused from the server\'s prompt cache']);
      if (s.total != null && s.tps == null) parts.push(['total', s.total.toFixed(2) + ' s']);
      if (v.stopped) parts.push(['', 'stopped']);
      else if (s.finish === 'length') parts.push(['', 'hit the token limit']);
      var row = el('div', 'stats mbx-mono');
      parts.forEach(function (p) { var sp = el('span', null); if (p[2]) sp.title = p[2]; if (p[0]) sp.appendChild(el('i', null, p[0] + ' ')); sp.appendChild(document.createTextNode(p[1])); row.appendChild(sp); });
      return row;
    }
    function botEl(c, m, i) {
      var v = m.versions[m.sel || 0], last = i === c.messages.length - 1, live = busy && busy.msg === m;
      var d = el('div', 'msg bot' + (live ? ' live' : '')); d.dataset.i = i;
      var head = el('div', 'who');
      head.appendChild(el('span', 'mbx-pxl', c.model || model.id || 'model'));
      if (m.versions.length > 1) {
        var nav = el('span', 'vers mbx-mono');
        var p = el('button', 'sm ghost', '‹'), n = el('button', 'sm ghost', '›');
        p.type = n.type = 'button'; p.setAttribute('aria-label', 'Previous reply'); n.setAttribute('aria-label', 'Next reply');
        p.disabled = live || !m.sel; n.disabled = live || m.sel >= m.versions.length - 1;
        p.addEventListener('click', function () { m.sel--; persist(); drawLog(); drawCtx(); });
        n.addEventListener('click', function () { m.sel++; persist(); drawLog(); drawCtx(); });
        nav.appendChild(p); nav.appendChild(el('span', null, (m.sel + 1) + ' / ' + m.versions.length)); nav.appendChild(n);
        head.appendChild(nav);
      }
      d.appendChild(head);
      if (v.reasoning || (live && v.thinking)) {
        var det = el('details', 'think'); det.open = !!(live && !v.content);
        var sum = el('summary', null, v.content || !live ? 'Thinking' + (v.stats && v.stats.think != null ? ' · ' + v.stats.think.toFixed(1) + ' s' : '') : 'Thinking…');
        det.appendChild(sum);
        var tb = el('div', 'think-b'); tb.textContent = v.reasoning || ''; det.appendChild(tb);
        d.appendChild(det);
      }
      var body = el('div', 'md'); body.innerHTML = X.md(v.content || '');
      if (live && !v.content && !v.reasoning) body.appendChild(el('span', 'wait mbx-pxl', 'Waiting for the first token'));
      d.appendChild(body);
      if (v.error) {
        var er = el('div', 'err');
        er.appendChild(el('b', null, 'The server returned an error. '));
        er.appendChild(document.createTextNode(v.error));
        d.appendChild(er);
      }
      if (!live) {
        d.appendChild(statLine(v));
        d.appendChild(actions([
          ['Copy', function (e) { copy(v.content || '', e.currentTarget); }, 'Copy the reply as Markdown'],
          last ? [v.error ? 'Retry' : 'Regenerate', function () { regenerate(c); }, 'Write a new reply; earlier ones stay one click away'] : null
        ]));
      }
      return d;
    }

    // ---------------- sending
    function title(t) { t = t.replace(/\s+/g, ' ').trim(); return t.length > 60 ? t.slice(0, 57) + '...' : t; }
    function send() {
      var t = $('msg').value;
      if (!t.trim() || busy || !model.id) return;
      var c = cur(); if (!c) { newChat(); c = cur(); }
      c.system = $('system').value;
      if (!c.messages.length) c.title = title(t);
      c.messages.push({ role: 'user', content: t });
      c.model = model.id;
      $('msg').value = ''; grow();
      drawHistory(); drawLog(); reply(c);
    }
    function regenerate(c) {
      if (busy) return;
      var m = c.messages[c.messages.length - 1];
      if (!m || m.role !== 'assistant') return;
      c.messages.pop();
      reply(c, m);
    }
    function reply(c, prev) {
      var hist = X.toMessages(c, { keepThinking: caps.keep_thinking });
      var v = { content: '', reasoning: '', thinking: caps.thinking && !!sets.thinking };
      var m = prev || { role: 'assistant', versions: [], sel: 0 };
      if (prev && prev.versions[prev.sel] && prev.versions[prev.sel].error && !prev.versions[prev.sel].content) prev.versions.splice(prev.sel, 1);
      m.versions.push(v); m.sel = m.versions.length - 1;
      c.messages.push(m); c.updated = Date.now();
      var body = X.buildBody(model.id, hist, sets, caps);
      var ctrl = new AbortController(), t = { t0: performance.now(), tFirst: null, tThinkEnd: null, tEnd: null, usage: null, finish: null };
      busy = { chat: c.id, msg: m, ctrl: ctrl, v: v };
      buttons(); MBX.status('busy', 'Replying');
      var follow = true, node = null, raf = 0;
      drawLog(); node = $('log').lastChild; toBottom();
      function paint() {
        raf = 0;
        if (!node || !node.isConnected) return;
        follow = follow && nearBottom();
        var fresh = botEl(c, m, c.messages.length - 1);
        var det = node.querySelector('details.think'), fdet = fresh.querySelector('details.think');
        if (fdet && busy && busy.thinkOpen != null) fdet.open = busy.thinkOpen;
        node.replaceWith(fresh); node = fresh;
        var tb = node.querySelector('.think-b'); if (tb && det && det.open) tb.scrollTop = tb.scrollHeight;
        if (follow) toBottom();
      }
      function later() { if (!raf) raf = requestAnimationFrame(paint); }
      fetch('/v1/chat/completions', { method: 'POST', headers: hdr({ 'Content-Type': 'application/json' }), body: JSON.stringify(body), signal: ctrl.signal })
        .then(function (r) {
          if (!r.ok || !r.body) return r.text().then(function (txt) { var j = null; try { j = JSON.parse(txt); } catch (e) {} throw new Error((j && j.error && (j.error.message || j.error)) || ('HTTP ' + r.status + ' ' + txt.slice(0, 200))); });
          var rd = r.body.getReader(), dec = new TextDecoder(), feed = X.sseReader();
          function pump() {
            return rd.read().then(function (x) {
              var evs = feed(x.done ? '' : dec.decode(x.value, { stream: true }), x.done);
              evs.forEach(function (data) {
                if (data === '[DONE]') return;
                var j; try { j = JSON.parse(data); } catch (e) { return; }
                if (j.error) throw new Error(j.error.message || String(j.error));
                if (j.usage) t.usage = j.usage;
                if (j.timings) t.timings = j.timings;
                var ch = (j.choices || [])[0];
                if (!ch) return;
                var d = ch.delta || {}, rs = d.reasoning_content || d.reasoning || '', tx = d.content || '';
                if ((rs || tx) && t.tFirst == null) t.tFirst = performance.now();
                if (rs) v.reasoning += rs;
                if (tx) { if (v.reasoning && t.tThinkEnd == null) t.tThinkEnd = performance.now(); v.content += tx; }
                if (ch.finish_reason) t.finish = ch.finish_reason;
              });
              if (evs.length) later();
              if (!x.done) return pump();
            });
          }
          return pump();
        })
        .catch(function (e) {
          if (e && e.name === 'AbortError') v.stopped = true;
          else v.error = (e && e.message) || String(e);
        })
        .then(function () {
          t.tEnd = performance.now();
          if (v.reasoning && t.tThinkEnd == null && v.content === '' ) t.tThinkEnd = t.tEnd;
          v.stats = X.replyStats(t); delete v.thinking;
          busy = null; if (raf) cancelAnimationFrame(raf);
          c.updated = Date.now(); persist();
          follow = follow && nearBottom();
          drawLog(); drawHistory(); drawCtx(); buttons();
          if (follow) toBottom();
          MBX.status(v.error ? 'error' : 'live', v.error ? 'Last reply failed' : 'Ready');
          if (v.error) health();
        });
    }
    function stop() { if (busy) busy.ctrl.abort(); }
    function buttons() {
      $('send').hidden = !!busy; $('stop').hidden = !busy;
      $('send').disabled = !model.id;
      $('new-chat').disabled = !!busy;
    }

    // ---------------- the server
    var tries = 0;
    function health() {
      fetch('/v1/models', { cache: 'no-store', headers: hdr() }).then(function (r) {
        if (r.status === 503) throw new Error('loading');
        if (!r.ok) throw new Error('HTTP ' + r.status);
        return r.json();
      }).then(function (j) {
        var m = (j.data || [])[0] || {};
        var first = !model.id;
        model.id = m.id || MBX.config.model; model.ctx = m.max_model_len || m.context_length || m.max_context_length || null; model.info = m;
        tries = 0;
        if (first) { initSettings(); if (!cur()) newChat(); }
        MBX.status(busy ? 'busy' : 'live', busy ? 'Replying' : 'Ready');
        $('msg').placeholder = 'Message ' + (MBX.config.name || model.id);
        drawAll(); buttons(); serverInfo();
      }).catch(function (e) {
        tries++;
        MBX.status(e.message === 'loading' ? 'busy' : 'error', e.message === 'loading' ? 'Loading the model' : 'Server not reachable');
        setTimeout(health, Math.min(10000, 1000 * tries));
      });
      fetch('/health', { cache: 'no-store', headers: hdr() }).then(function (r) { if (r.status === 503) MBX.status('busy', 'Loading the model'); }).catch(function () {});
    }
    function serverInfo() {
      var dl = $('srv'); dl.textContent = '';
      var rows = [['Model', model.id], ['Context', model.ctx ? fmtN(model.ctx) + ' tokens' : null], ['Endpoint', location.origin + '/v1']];
      Object.keys(model.info || {}).forEach(function (k) {
        var v = model.info[k];
        if (['id', 'object', 'created', 'owned_by', 'max_model_len', 'root', 'parent', 'permission'].indexOf(k) >= 0 || v == null || typeof v === 'object') return;
        rows.push([k.replace(/_/g, ' '), String(v)]);
      });
      rows.forEach(function (r) { if (r[1] == null) return; dl.appendChild(el('dt', null, r[0])); dl.appendChild(el('dd', 'mbx-mono', r[1])); });
      apiCode();
    }
    var lang = 'curl';
    function apiCode() {
      if (!sets || !$('call')) return;
      var c = cur(), msgs = c && c.messages.length ? X.toMessages(c, { keepThinking: caps.keep_thinking }) : [{ role: 'user', content: 'Hello' }];
      if (!(c && c.messages.length) && $('system').value.trim()) msgs.unshift({ role: 'system', content: $('system').value });
      var b = X.buildBody(model.id || MBX.config.model, msgs, sets, caps);
      delete b.stream_options; b.stream = false;
      var base = location.origin + '/v1', code;
      if (lang === 'curl') code = 'curl ' + base + '/chat/completions \\\n  -H "Content-Type: application/json" \\\n  -d \'' + JSON.stringify(b, null, 2).replace(/'/g, "'\\''") + '\'';
      else if (lang === 'python') {
        var extra = {}, std = ['model', 'messages', 'stream', 'temperature', 'top_p', 'max_tokens', 'seed', 'stop', 'presence_penalty', 'frequency_penalty'];
        Object.keys(b).forEach(function (k) { if (std.indexOf(k) < 0) extra[k] = b[k]; });
        code = 'from openai import OpenAI\n\nclient = OpenAI(base_url="' + base + '", api_key="none")\nr = client.chat.completions.create(\n' +
          std.filter(function (k) { return b[k] != null && k !== 'stream'; }).map(function (k) { return '    ' + k + '=' + py(b[k]) + ','; }).join('\n') +
          (Object.keys(extra).length ? '\n    extra_body=' + py(extra) + ',' : '') + '\n)\nprint(r.choices[0].message.content)';
      }
      $('call').textContent = code;
    }
    function py(v) { return JSON.stringify(v, null, 4).replace(/\n/g, '\n    ').replace(/\btrue\b/g, 'True').replace(/\bfalse\b/g, 'False').replace(/\bnull\b/g, 'None'); }

    // ---------------- export and import
    function download(name, text, type) {
      var a = document.createElement('a'); a.href = URL.createObjectURL(new Blob([text], { type: type })); a.download = name;
      document.body.appendChild(a); a.click(); setTimeout(function () { URL.revokeObjectURL(a.href); a.remove(); }, 500);
    }
    function slug(c) { return ((c.title || 'chat').toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '').slice(0, 40) || 'chat'); }
    function exportChat(kind) {
      var c = cur(); if (!c || !c.messages.length) return;
      if (kind === 'md') return download(slug(c) + '.md', X.chatMarkdown(c, model.id), 'text/markdown');
      download(slug(c) + '.json', JSON.stringify({ format: 'mashbox-chat', version: 1, model: c.model, title: c.title, created: new Date(c.created).toISOString(),
        settings: sets, messages: X.toMessages(c, { keepThinking: true }), chat: c }, null, 2), 'application/json');
    }
    function importChats(file) {
      var r = new FileReader();
      r.onload = function () {
        try {
          var j = JSON.parse(r.result), list = Array.isArray(j) ? j : [j], n = 0;
          list.forEach(function (x) {
            var c = x.chat || null;
            if (!c && Array.isArray(x.messages)) {
              c = { title: x.title || '', model: x.model || model.id, system: '', messages: [] };
              x.messages.forEach(function (m) {
                if (m.role === 'system') c.system = m.content;
                else if (m.role === 'user') c.messages.push({ role: 'user', content: String(m.content || '') });
                else if (m.role === 'assistant') c.messages.push({ role: 'assistant', sel: 0, versions: [{ content: String(m.content || ''), reasoning: m.reasoning_content || '' }] });
              });
            }
            if (!c || !Array.isArray(c.messages)) return;
            c.id = uid(); c.created = c.created || Date.now(); c.updated = Date.now();
            c.title = c.title || (c.messages[0] ? title(c.messages[0].content || '') : 'Imported chat');
            state.chats.unshift(c); state.cur = c.id; n++;
          });
          persist(); drawAll(); alert(n ? 'Imported ' + n + ' chat' + (n > 1 ? 's' : '') + '.' : 'No chat found in that file.');
        } catch (e) { alert('That file is not a chat export: ' + e.message); }
      };
      r.readAsText(file);
    }

    // ---------------- composer, drawers, shortcuts
    function grow() { var t = $('msg'); t.style.height = 'auto'; t.style.height = Math.min(t.scrollHeight + 4, Math.round(window.innerHeight * 0.4)) + 'px'; }
    function drawer(which) {
      ['side', 'set'].forEach(function (id) { $(id).classList.toggle('open', id === which); });
      $('scrim').hidden = !which;
      document.body.classList.toggle('mbx-drawer', !!which);
    }
    function typing(e) { var t = e.target; return t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.tagName === 'SELECT' || t.isContentEditable); }
    function keys(e) {
      var mod = e.ctrlKey || e.metaKey;
      if (e.key === 'Escape') {
        if (busy) { e.preventDefault(); stop(); return; }
        if ($('keys-dlg').open) return;
        if (document.body.classList.contains('mbx-drawer')) { drawer(null); return; }
      }
      if (mod && e.shiftKey && (e.key === 'o' || e.key === 'O')) { e.preventDefault(); newChat(true); return; }
      if (mod && e.shiftKey && e.key === 'Enter') { e.preventDefault(); var c = cur(); if (c) regenerate(c); return; }
      if (typing(e) || mod || e.altKey) return;
      if (e.key === '/') { e.preventDefault(); $('msg').focus(); }
      else if (e.key === '?') { e.preventDefault(); $('keys-dlg').showModal(); }
    }

    function start() {
      if (started) return; started = true;
      var blk = $('mbx-chat');
      try { Object.assign(caps, JSON.parse(blk ? blk.textContent : '{}') || {}); } catch (e) {}
      restore();
      sets = Object.assign({ preset: 'server', thinking: caps.thinking_default, max_tokens: null, seed: null, stop: '', system: '' }, GENERIC);
      buildSettings();
      $('composer').addEventListener('submit', function (e) { e.preventDefault(); send(); });
      $('msg').addEventListener('input', grow);
      $('msg').addEventListener('keydown', function (e) {
        if (e.key === 'Enter' && !e.shiftKey && !e.isComposing && !(e.ctrlKey || e.metaKey) && !('ontouchstart' in window && window.matchMedia('(pointer: coarse)').matches)) { e.preventDefault(); send(); }
        if (e.key === 'ArrowUp' && !$('msg').value) {
          var c = cur(); if (!c || busy) return;
          for (var i = c.messages.length - 1; i >= 0; i--) if (c.messages[i].role === 'user') {
            e.preventDefault(); var d = $('log').querySelector('.msg[data-i="' + i + '"]'); if (d) { editUser(c, i, d); d.scrollIntoView({ block: 'center' }); } break;
          }
        }
      });
      $('stop').addEventListener('click', stop);
      $('new-chat').addEventListener('click', function () { newChat(true); });
      $('think').addEventListener('click', function (e) {
        var b = e.target.closest('button[data-think]'); if (!b) return;
        sets.thinking = b.dataset.think === '1';
        if (sets.preset === 'rec') applyPreset('rec'); else { keepSets(); thinkMarks(); apiCode(); }
        if (cur() && !cur().messages.length) drawLog();
      });
      $('find').addEventListener('input', drawHistory);
      $('open-chats').addEventListener('click', function () { drawer($('side').classList.contains('open') ? null : 'side'); });
      $('open-set').addEventListener('click', function () { drawer($('set').classList.contains('open') ? null : 'set'); });
      $('scrim').addEventListener('click', function () { drawer(null); });
      document.querySelectorAll('[data-close]').forEach(function (b) { b.addEventListener('click', function () { drawer(null); }); });
      $('export-md').addEventListener('click', function () { exportChat('md'); });
      $('export-json').addEventListener('click', function () { exportChat('json'); });
      $('import').addEventListener('click', function () { $('import-file').click(); });
      $('import-file').addEventListener('change', function () { if (this.files[0]) importChats(this.files[0]); this.value = ''; });
      $('clear-all').addEventListener('click', function () {
        if (busy || !state.chats.some(function (c) { return c.messages.length; })) return;
        if (!confirm('Delete every chat saved in this browser?')) return;
        state.chats = []; state.cur = null; persist(); newChat(); drawAll();
      });
      $('lang').addEventListener('click', function (e) { var b = e.target.closest('button[data-lang]'); if (!b) return; lang = b.dataset.lang; $('lang').querySelectorAll('button').forEach(function (x) { x.classList.toggle('on', x === b); }); apiCode(); });
      $('copy-call').addEventListener('click', function (e) { copy($('call').textContent, e.currentTarget); });
      $('keys').addEventListener('click', function () { $('keys-dlg').showModal(); });
      $('keys-close').addEventListener('click', function () { $('keys-dlg').close(); });
      $('log').addEventListener('click', function (e) {
        var s = e.target.closest('details.think > summary');
        if (s && busy && s.closest('.msg.live')) busy.thinkOpen = !s.parentNode.open;   // the reader's choice wins while it streams
        var b = e.target.closest('[data-copy]'); if (!b) return;
        var code = b.closest('.cb').querySelector('code'); copy(code.textContent, b);
      });
      document.addEventListener('keydown', keys);
      var mq = window.matchMedia('(pointer: coarse)');
      $('hint').textContent = mq.matches ? 'Tap Send to send' : 'Enter sends · Shift+Enter new line · ? shortcuts';
      buttons(); drawAll(); health();
    }
    document.addEventListener('mbx:ready', start);
    if (document.readyState !== 'loading') setTimeout(start, 0); else document.addEventListener('DOMContentLoaded', function () { setTimeout(start, 0); });
  }
});
