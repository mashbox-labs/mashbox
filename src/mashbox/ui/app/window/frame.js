/* The MASHBOX window: a sidebar of modules (one per model kind the app knows) and app screens, and one page at a time
   in the frame. Cmd/Ctrl + 1..9 picks an entry. Links to the web open in the default browser. */
(function () {
  'use strict';
  var MODULES = [{ id: 'chat', label: 'Chat', href: '/m/chat/' }, { id: 'world', label: 'World model', href: '/m/world/' }];
  var TOOLS = [{ id: 'models', label: 'Models', href: '/p/models' }, { id: 'servers', label: 'Servers', href: '/p/servers' },
    { id: 'agents', label: 'Connect agent', href: '/p/agents' }, { id: 'settings', label: 'Settings', href: '/p/settings' }];
  var ALL = MODULES.concat(TOOLS);
  var frame = document.getElementById('view');
  var KEY = 'mbx-app-view';
  // this launch's token: the app server wants it on every /api call (the page carries it; other sites cannot read it)
  var TOKEN = (document.querySelector('meta[name="mbx-token"]') || {}).content || '';
  function hdr(h) { h = h || {}; if (TOKEN) h['X-Mashbox-Token'] = TOKEN; return h; }
  function nav(box, items, off) {
    items.forEach(function (it, i) {
      var a = document.createElement('a');
      a.href = it.href; a.dataset.id = it.id;
      a.appendChild(document.createTextNode(it.label));
      var k = document.createElement('kbd'); k.textContent = '⌘' + (off + i + 1); a.appendChild(k);
      a.addEventListener('click', function (e) { e.preventDefault(); open(it.id); });
      box.appendChild(a);
    });
  }
  nav(document.getElementById('mods'), MODULES, 0);
  nav(document.getElementById('tools'), TOOLS, MODULES.length);
  function mark(id) {
    Array.prototype.forEach.call(document.querySelectorAll('.app-nav a'), function (a) { a.classList.toggle('sel', a.dataset.id === id); });
  }
  function open(id) {
    var it = ALL.filter(function (x) { return x.id === id; })[0] || ALL[0];
    frame.src = it.href; mark(it.id);
    try { localStorage.setItem(KEY, it.id); } catch (e) {}
  }
  function idOf(path) {
    if (/^\/m\/chat/.test(path)) return 'chat';
    if (/^\/m\/world|^\/p\/world/.test(path)) return 'world';
    var m = path.match(/^\/p\/(\w+)/); return m ? m[1] : null;
  }
  frame.addEventListener('load', function () {
    var doc;
    try { doc = frame.contentDocument; } catch (e) { doc = null; }      // the world model page: its own origin
    if (!doc) return;
    var id = idOf(frame.contentWindow.location.pathname);
    if (id) { mark(id); try { localStorage.setItem(KEY, id); } catch (e) {} }
    doc.addEventListener('click', function (e) {
      var a = e.target.closest && e.target.closest('a[href]');
      if (!a) return;
      var u;
      try { u = new URL(a.getAttribute('href'), frame.contentWindow.location.href); } catch (err) { e.preventDefault(); return; }
      // only http(s) links are followed: mailto:, javascript:, data: and the rest never load in the window
      if (!/^https?:$/.test(u.protocol)) { e.preventDefault(); return; }
      if (u.origin !== location.origin) {
        e.preventDefault();
        fetch('/api/open', { method: 'POST', headers: hdr({ 'Content-Type': 'application/json' }), body: JSON.stringify({ url: u.href }) });
      } else if (/^\/(v1|api)(\/|$)|^\/health$/.test(u.pathname)) {
        e.preventDefault();                   // the server's answers are data, never a page shown here
      } else if (a.target === '_top' || a.target === '_blank') {
        e.preventDefault(); var t = idOf(u.pathname); if (t) open(t); else frame.src = u.pathname + u.search;
      }
    }, true);
    doc.addEventListener('keydown', keys);
  });
  function keys(e) {
    if ((e.metaKey || e.ctrlKey) && /^[1-9]$/.test(e.key) && ALL[+e.key - 1]) { e.preventDefault(); open(ALL[+e.key - 1].id); }
  }
  document.addEventListener('keydown', keys);
  window.addEventListener('storage', function (e) { if (e.key === 'mbx-theme' && window.MBX) MBX.theme(e.newValue); });
  var srv = document.getElementById('srv');
  srv.addEventListener('click', function () { open('servers'); });
  function poll() {
    fetch('/api/state', { cache: 'no-store', headers: hdr() }).then(function (r) { return r.json(); }).then(function (s) {
      var x = (s.servers || []).filter(function (v) { return v.id === s.active; })[0];
      document.getElementById('srv-name').textContent = x ? x.name : 'None';
      document.getElementById('srv-model').textContent = x ? (x.model || x.url) : 'Add one in Servers';
      document.getElementById('srv-dot').className = 'app-dot ' + (x ? 'live' : 'off');
    }).catch(function () {});
  }
  poll(); setInterval(poll, 5000);
  var start = null;
  try { start = localStorage.getItem(KEY); } catch (e) {}
  fetch('/api/state', { cache: 'no-store', headers: hdr() }).then(function (r) { return r.json(); }).then(function (s) {
    open(start || (s.active ? 'chat' : 'servers'));
  }).catch(function () { open('servers'); });
})();
