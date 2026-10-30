// The MASHBOX app's window (src/mashbox/ui/app/window/frame.js) and its Connect-agent screen (src/mashbox/ui/app/view.js)
// run unchanged in Node on a small fake DOM, so tests/cpu/test_app_security.py checks the real files.
//   node app_js_checks.js frame <frame.js> '["href", ...]'          -> [{href, src, prevented, opened}] per link click
//   node app_js_checks.js agents <view.js> '{"state": ..., "check": ...}' -> [{title, text}] for every copy box
//   node app_js_checks.js servers <view.js> '{"state": ..., "check": ...}' -> [{id, acts}] per server row (its buttons)
'use strict';
const fs = require('fs');
const vm = require('vm');
const [, , mode, file, input] = process.argv;
const code = fs.readFileSync(file, 'utf8');
const ORIGIN = 'http://127.0.0.1:18930';

function node(tag) {
  const n = {
    tagName: String(tag).toUpperCase(), attrs: {}, children: [], listeners: {}, dataset: {}, style: {}, parent: null,
    classList: { add() {}, remove() {}, toggle() {}, contains() { return false; } },
    setAttribute(k, v) { this.attrs[k] = String(v); }, getAttribute(k) { return this.attrs[k]; },
    appendChild(c) { c.parent = this; this.children.push(c); return c; },
    insertBefore(c, ref) { const i = this.children.indexOf(ref); c.parent = this; this.children.splice(i < 0 ? this.children.length : i, 0, c); return c; },
    append(...cs) { cs.forEach((c) => this.appendChild(typeof c === 'string' ? text(c) : c)); },
    remove() { if (this.parent) this.parent.children = this.parent.children.filter((c) => c !== this); },
    addEventListener(e, f) { (this.listeners[e] = this.listeners[e] || []).push(f); },
    querySelector() { return null; }, querySelectorAll() { return []; }, select() {}, focus() {},
    get textContent() { return this.children.map((c) => c.textContent).join(''); },
    set textContent(v) { this.children = v ? [text(v)] : []; },
  };
  return n;
}
function text(t) { return { nodeType: 3, data: String(t), get textContent() { return this.data; } }; }
function walk(n, f) { f(n); (n.children || []).forEach((c) => walk(c, f)); }
const ids = {};
const document = {
  createElement: node, createTextNode: text, body: node('body'),
  getElementById(id) { return (ids[id] = ids[id] || node('div')); },
  querySelector(sel) { return /mbx-token/.test(sel) ? { content: 'launch-token' } : null; },
  querySelectorAll() { return []; }, addEventListener() {}, execCommand() { return true; },
};
const fetches = [];
let answers = {};
function fetch(u, opts) {
  fetches.push({ url: String(u), opts: opts || {} });
  const path = String(u).split('?')[0];
  const body = Object.prototype.hasOwnProperty.call(answers, path) ? answers[path] : {};
  return Promise.resolve({ status: 200, json: () => Promise.resolve(body), text: () => Promise.resolve(JSON.stringify(body)) });
}
const sandbox = {
  document, fetch, URL, URLSearchParams, Promise, JSON, Object, Array, String, Number, Math, Date, RegExp, console,
  setTimeout: () => 0, setInterval: () => 0, clearTimeout() {}, localStorage: { getItem: () => null, setItem() {} },
  addEventListener() {}, navigator: {}, location: { origin: ORIGIN, pathname: '/', search: '', href: ORIGIN + '/' },
};
sandbox.window = sandbox; sandbox.self = sandbox;
const tick = () => new Promise((r) => setImmediate(r));

(async () => {
  if (mode === 'frame') {
    const frame = ids.view = node('iframe');
    const doc = node('document');
    frame.contentDocument = doc;
    frame.contentWindow = { location: { pathname: '/m/chat/', href: ORIGIN + '/m/chat/' } };
    answers = { '/api/state': { servers: [], active: null } };
    vm.runInNewContext(code, sandbox);
    for (let i = 0; i < 5; i++) await tick();
    frame.listeners.load.forEach((f) => f());
    const out = [];
    for (const href of JSON.parse(input)) {
      frame.src = 'BEFORE';
      const before = fetches.length;
      let prevented = false;
      const a = { target: '_blank', getAttribute: () => href };
      const ev = { target: { closest: () => a }, preventDefault() { prevented = true; } };
      (doc.listeners.click || []).forEach((f) => f(ev));
      out.push({ href, src: frame.src, prevented, opened: fetches.slice(before).filter((x) => x.url === '/api/open').map((x) => JSON.parse(x.opts.body).url) });
    }
    console.log(JSON.stringify(out));
  } else if (mode === 'agents') {
    const inp = JSON.parse(input);
    answers = { '/api/state': inp.state, '/api/check': inp.check };
    sandbox.location.pathname = '/p/agents';
    sandbox.location.href = ORIGIN + '/p/agents';
    vm.runInNewContext(code, sandbox);
    for (let i = 0; i < 10; i++) await tick();
    const boxes = [];
    walk(ids['app-view'], (n) => {
      if (n.attrs && n.attrs.class === 'app-copy') {
        const head = n.children[0], pre = n.children[1];
        boxes.push({ title: head.children[0].textContent, text: pre.textContent });
      }
    });
    console.log(JSON.stringify(boxes));
  } else if (mode === 'servers') {
    const inp = JSON.parse(input);
    answers = { '/api/state': inp.state, '/api/check': inp.check, '/api/catalog': { models: [] }, '/api/gcloud': {} };
    sandbox.location.pathname = '/p/servers';
    sandbox.location.href = ORIGIN + '/p/servers';
    vm.runInNewContext(code, sandbox);
    for (let i = 0; i < 10; i++) await tick();
    const rows = [];
    walk(ids['app-view'], (n) => {
      if (n.attrs && n.attrs['data-server']) {
        const acts = [];
        walk(n, (m) => { if (m.tagName === 'BUTTON' || m.tagName === 'A') acts.push(m.textContent); });
        rows.push({ id: n.attrs['data-server'], acts });
      }
    });
    console.log(JSON.stringify(rows));
  } else {
    throw new Error('mode is frame, agents or servers');
  }
})().catch((e) => { console.error(e && e.stack || e); process.exit(1); });
