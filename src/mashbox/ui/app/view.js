/* MASHBOX app screens: Models, Servers, Connect your agent, Settings and the World model start screen. Each is a
   module shell page; this file fills its content area from the app server's /api calls. Text from servers and the
   catalog always goes in as text, never as markup. */
(function () {
  'use strict';
  var view = (location.pathname.match(/\/p\/(\w+)/) || [])[1] || 'servers';
  var params = new URLSearchParams(location.search);
  var main = document.getElementById('app-view');
  var DOCS = 'https://mashbox.gg/docs/';
  var TOKEN = (document.querySelector('meta[name="mbx-token"]') || {}).content || '';   // this launch's, for /api
  function hdr(h) { h = h || {}; if (TOKEN) h['X-Mashbox-Token'] = TOKEN; return h; }

  function el(tag, attrs, kids) {
    var n = document.createElement(tag);
    Object.keys(attrs || {}).forEach(function (k) {
      if (k === 'on') Object.keys(attrs.on).forEach(function (e) { n.addEventListener(e, attrs.on[e]); });
      else if (attrs[k] !== null && attrs[k] !== undefined && attrs[k] !== false) n.setAttribute(k, attrs[k] === true ? '' : attrs[k]);
    });
    (kids || []).forEach(function (c) { if (c !== null && c !== undefined) n.appendChild(typeof c === 'string' ? document.createTextNode(c) : c); });
    return n;
  }
  function get(u) { return fetch(u, { cache: 'no-store', headers: hdr() }).then(function (r) { return r.json(); }); }
  function post(u, b) {
    return fetch(u, { method: 'POST', headers: hdr({ 'Content-Type': 'application/json' }), body: JSON.stringify(b || {}) })
      .then(function (r) { return r.json().catch(function () { return {}; }); });
  }
  function panel(title, kids, cls) { return el('section', { class: 'mbx-panel app-panel ' + (cls || '') }, [el('h3', {}, [title])].concat(kids)); }
  function note(t, cls) { return el('p', { class: 'mbx-note ' + (cls || '') }, [t]); }
  function field(label, input, hint) {
    return el('label', { class: 'mbx-field app-field' }, [el('span', { class: 'mbx-label' }, [label]), input, hint ? el('small', { class: 'app-hint' }, [hint]) : null]);
  }
  function copyBox(title, text, where) {
    var pre = el('pre', { class: 'app-code' }, [text]);
    var b = el('button', { class: 'sm', type: 'button' }, ['Copy']);
    b.addEventListener('click', function () {
      var done = function () { b.textContent = 'Copied'; setTimeout(function () { b.textContent = 'Copy'; }, 1400); };
      if (navigator.clipboard && navigator.clipboard.writeText) navigator.clipboard.writeText(text).then(done, function () { fallback(); });
      else fallback();
      function fallback() { var t = el('textarea', {}, [text]); document.body.appendChild(t); t.select(); try { document.execCommand('copy'); done(); } catch (e) {} t.remove(); }
    });
    return el('div', { class: 'app-copy' }, [el('div', { class: 'app-copy-head' }, [el('b', {}, [title]), where ? el('span', { class: 'app-dim' }, [where]) : null, b]), pre]);
  }
  function link(url, text) { return el('a', { href: url, target: '_blank', rel: 'noopener' }, [text]); }
  function status(state, text) { if (window.MBX && MBX.status) MBX.status(state, text); }
  function plain(s) { return typeof s === 'string' && /^[\x21\x23-\x5b\x5d-\x7e]{1,200}$/.test(s); }   // printable ASCII, no quote or backslash
  function toml(s) {               // a TOML basic string: quotes, backslashes and control characters escaped
    return '"' + String(s).replace(/[\\"\u0000-\u001f\u007f]/g, function (c) {
      return c === '\\' ? '\\\\' : c === '"' ? '\\"' : '\\u' + ('000' + c.charCodeAt(0).toString(16)).slice(-4);
    }) + '"';
  }
  function when(t) { if (!t) return 'never'; var d = new Date(t * 1000); return d.toLocaleString(); }

  // ------------------------------------------------------------------------------------------------ Models
  function models() {
    main.textContent = '';
    var list = el('div', { class: 'app-list' });
    var src = el('p', { class: 'mbx-note' }, ['Loading the catalog']);
    var refresh = el('button', { class: 'sm', type: 'button', on: { click: function () { load(true); } } }, ['Refresh']);
    main.appendChild(panel('Models', [el('div', { class: 'app-row' }, [src, refresh]), list]));
    function load(r) {
      get('/api/catalog' + (r ? '?refresh=1' : '')).then(function (c) {
        var s = c.source || {};
        var sg = s.signed ? ', signed by MASHBOX Labs' + (s.signed_utc ? ' ' + s.signed_utc : '') : ', not signed (your own catalog address)';
        src.textContent = s.from === 'url' ? 'Catalog read ' + when(s.at) + ' from ' + s.url + sg
          : s.from === 'cache' ? 'Using the catalog saved ' + when(s.at) + sg + ' (' + (s.error || '') + ')'
          : 'Offline: the catalog in this install (' + (s.error || '') + ')';
        list.textContent = '';
        c.models.forEach(function (m) {
          var acts = [];
          if (m.module === 'chat' && m.served) acts.push(el('a', { class: 'app-btn pri', href: '/m/chat/', target: '_top', 'data-goto': 'chat' }, ['Chat']));
          if (m.module === 'world' && m.served) acts.push(el('a', { class: 'app-btn pri', href: '/m/world/', target: '_top' }, ['Open']));
          if (m.tpu && m.module) acts.push(el('a', { class: 'app-btn', href: '/p/servers?model=' + encodeURIComponent(m.id) }, ['Run on a TPU']));
          var tags = [el('span', { class: 'mbx-badge' }, [m.module ? (m.module === 'chat' ? 'Chat' : 'World model') : m.kind])]
            .concat((m.chips || []).map(function (c) { return el('span', { class: 'mbx-chip' }, [c]); }));
          if (m.served) tags.push(el('span', { class: 'mbx-chip app-live' }, ['On your server']));
          if (m.launch) tags.push(el('span', { class: 'mbx-chip' }, [m.launch]));
          list.appendChild(el('article', { class: 'app-item', 'data-model': m.id }, [
            el('div', { class: 'app-item-main' }, [
              el('b', { class: 'app-name' }, [m.name]), m.maker ? el('span', { class: 'app-dim' }, [' from ' + m.maker]) : null,
              el('p', {}, [m.summary || '']), el('div', { class: 'app-tags' }, tags),
              m.module ? null : note('This kind of model needs a newer MASHBOX app.')]),
            el('div', { class: 'app-acts' }, acts)]));
        });
        if (!c.models.length) list.appendChild(note('The catalog lists no models.'));
      }).catch(function (e) { src.textContent = 'Could not read the catalog: ' + e; });
    }
    load(false);
  }

  // ------------------------------------------------------------------------------------------------ Servers
  function servers() {
    main.textContent = '';
    var mine = el('div', { class: 'app-list' });
    var name = el('input', { placeholder: 'My server', autocomplete: 'off' });
    var url = el('input', { placeholder: 'http://127.0.0.1:8000', autocomplete: 'off', spellcheck: 'false' });
    var key = el('input', { type: 'password', placeholder: 'only if the server asks for one', autocomplete: 'off' });
    var msg = note('');
    var add = el('button', { class: 'pri', type: 'button' }, ['Add and use']);
    var test = el('button', { type: 'button' }, ['Test']);
    test.addEventListener('click', function () {
      msg.textContent = 'Checking ' + url.value;
      post('/api/servers/check', { url: url.value, key: key.value }).then(function (p) {
        msg.textContent = p.ok ? 'Answered in ' + p.ms + ' ms' + (p.models.length ? ': ' + p.models.map(function (m) { return m.id; }).join(', ') : '') : (p.error || 'No answer');
      });
    });
    add.addEventListener('click', function () {
      post('/api/servers/add', { name: name.value, url: url.value, key: key.value }).then(function (r) {
        if (r.error) { msg.textContent = r.error; return; }
        msg.textContent = r.probe && r.probe.ok ? 'Added.' : 'Added, but it did not answer: ' + ((r.probe && r.probe.error) || '');
        name.value = url.value = key.value = ''; draw();
      });
    });
    main.appendChild(el('div', { class: 'mbx-cols app-cols' }, [
      el('div', {}, [panel('Your servers', [mine]),
        panel('Add a server', [note('Any MASHBOX server or OpenAI-compatible server: its address, the part before /v1.'),
          field('Name', name), field('Address', url), field('Key', key, 'Kept on this Mac only, in a file only you can read.'),
          el('div', { class: 'app-row' }, [add, test]), msg])]),
      tpuPanel()]));
    function draw() {
      get('/api/state').then(function (s) {
        mine.textContent = '';
        if (!s.servers.length) mine.appendChild(note('None. Start a TPU or add a server address.'));
        s.servers.forEach(function (x) {
          var on = x.id === s.active;
          var light = el('span', { class: 'app-dot' });
          var info = el('small', { class: 'app-dim' }, ['checking']);
          get('/api/check?id=' + encodeURIComponent(x.id)).then(function (p) {
            light.className = 'app-dot ' + (p.ok ? (p.health === 503 ? 'busy' : 'live') : 'off');
            info.textContent = p.ok ? (p.health === 503 ? 'loading the model' : 'answers in ' + p.ms + ' ms') + (p.models.length ? ' · ' + p.models.map(function (m) { return m.id; }).join(', ') : '') : (p.error || 'no answer');
          });
          var ext = (s.addons || []).filter(function (a) { return a.id === x.module; })[0];   // a server of an add-on module
          mine.appendChild(el('article', { class: 'app-item' + (on ? ' app-on' : ''), 'data-server': x.id }, [
            el('div', { class: 'app-item-main' }, [light, el('b', { class: 'app-name' }, [x.name]), on ? el('span', { class: 'mbx-chip app-live' }, ['In use']) : null,
              ext ? el('span', { class: 'mbx-chip' }, [ext.label]) : null,
              el('div', { class: 'app-mono' }, [x.url]), info]),
            el('div', { class: 'app-acts' }, [
              ext ? el('a', { class: 'app-btn pri', href: '/m/' + ext.id + '/', target: '_top' }, ['Open ' + ext.label]) :
              on ? null : el('button', { class: 'sm', type: 'button', on: { click: function () { post('/api/servers/use', { id: x.id }).then(draw); } } }, ['Use']),
              el('button', { class: 'sm ghost', type: 'button', on: { click: function () { post('/api/servers/remove', { id: x.id }).then(draw); } } }, ['Remove'])])]));
        });
      });
    }
    draw();
  }

  function tpuPanel() {
    var box = el('div');
    var gc = note('Checking Google Cloud');
    var project = el('input', { placeholder: 'your Google Cloud project id', autocomplete: 'off', spellcheck: 'false' });
    var model = el('select');
    var zone = el('input', { placeholder: 'any of the model\'s zones', autocomplete: 'off' });
    var chipName = el('input', { placeholder: 'mashbox-<model>', autocomplete: 'off', spellcheck: 'false' });
    var hours = el('input', { type: 'number', min: '0.5', step: '0.5', value: '2' });
    var spot = el('input', { type: 'checkbox' });
    var agents = el('input', { type: 'checkbox' });
    var price = note('');
    var start = el('button', { class: 'pri', type: 'button' }, ['Start']);
    var stop = el('button', { type: 'button' }, ['Close tunnel']);
    var down = el('button', { class: 'app-danger', type: 'button' }, ['Delete chip']);
    var reconnect = el('button', { type: 'button' }, ['Reconnect']);
    var log = el('pre', { class: 'app-log', 'aria-label': 'Output' }, ['']);
    var chips = el('pre', { class: 'app-log app-status' }, ['']);
    var signin = el('button', { class: 'sm', type: 'button', on: { click: function () { post('/api/gcloud/login').then(function () { gc.textContent = 'A browser window opened to sign in. Come back here when it says you are signed in.'; }); } } }, ['Sign in to Google Cloud']);
    var cat = {};
    get('/api/catalog').then(function (c) {
      var want = params.get('model');
      c.models.filter(function (m) { return m.tpu && m.module; })
        .sort(function (a, b) { return (a.module === 'chat' ? 0 : 1) - (b.module === 'chat' ? 0 : 1); }).forEach(function (m) {
        cat[m.id] = m;
        model.appendChild(el('option', { value: m.id, selected: want === m.id }, [m.name + ' (TPU ' + (m.accelerator || '') + ')']));
      });
      pick();
    });
    function pick() { var m = cat[model.value]; agents.disabled = !(m && m.agents); price.textContent = m ? 'Runs on a TPU ' + m.accelerator + '. The price per hour shows below when it starts, and in Status.' : ''; }
    model.addEventListener('change', function () { pick(); poll(); });
    get('/api/state').then(function (s) { project.value = s.project || ''; });
    get('/api/gcloud').then(function (g) {
      if (!g.gcloud) { gc.textContent = 'Google Cloud CLI (gcloud) is not installed. Install it from cloud.google.com/sdk, then reopen this screen.'; return; }
      gc.textContent = g.account ? 'Signed in to Google Cloud as ' + g.account + '.' : 'Not signed in to Google Cloud.';
      if (!project.value && g.project) project.value = g.project;
    });
    project.addEventListener('change', function () { post('/api/settings', { project: project.value }); });
    start.addEventListener('click', function () {
      post('/api/settings', { project: project.value }).then(function () {
        return post('/api/tpu/up', { model: model.value, zone: zone.value, name: chipName.value, max_hours: hours.value, spot: spot.checked, agents: agents.checked });
      }).then(show);
    });
    stop.addEventListener('click', function () { post('/api/tpu/stop', { model: model.value }).then(show); });
    reconnect.addEventListener('click', function () { post('/api/tpu/tunnel', { name: (job && job.chip) || '', model: model.value }).then(show); });
    // window.confirm shows nothing in the app's web view, so the button asks on itself: a second click within 5 s deletes
    var armed = 0;
    down.addEventListener('click', function () {
      var name = (job && job.chip) || '';
      if (Date.now() - armed > 5000) {
        armed = Date.now(); down.textContent = 'Click again to delete ' + (name || 'the chip');
        setTimeout(function () { if (armed && Date.now() - armed >= 5000) { armed = 0; down.textContent = 'Delete chip'; } }, 5100);
        return;
      }
      armed = 0; down.textContent = 'Delete chip';
      chips.textContent = 'Deleting ' + name + ' ... (up to a few minutes)';
      down.disabled = true;
      post('/api/tpu/down', { name: name, model: model.value }).then(function (r) { chips.textContent = r.out || ''; })
        .finally(function () { down.disabled = false; });
    });
    var job = null;
    function show(j) {
      if (j && j.error) { log.textContent = j.error; return; }
      job = j;
      start.disabled = !!(j && j.running);
      stop.disabled = !(j && j.running);
      log.textContent = j ? '$ ' + j.cmd + '\n' + j.lines.join('\n') : '';
      log.scrollTop = log.scrollHeight;
      if (j && j.url) status('live', 'Tunnel open: ' + j.url);
    }
    function poll() { get('/api/tpu/job?model=' + encodeURIComponent(model.value || '')).then(show).catch(function () {}); }
    function refreshStatus() {
      chips.textContent = 'Asking Google Cloud ...';
      get('/api/tpu/status').then(function (r) { chips.textContent = r.out || '(no output)'; });
    }
    poll(); setInterval(poll, 2500);
    var st = el('button', { class: 'sm', type: 'button', on: { click: refreshStatus } }, ['Status']);
    box.appendChild(panel('Your Cloud TPU', [
      note('MASHBOX creates a chip in your own Google Cloud project, installs the model, starts its server and opens a private tunnel to this Mac. Google bills your account by the hour while the chip exists.'),
      el('div', { class: 'app-row' }, [gc, signin]),
      field('Project', project), field('Model', model), price,
      el('div', { class: 'app-grid2' }, [field('Zone', zone), field('Deletes itself after (hours)', hours)]),
      field('Chip name', chipName, 'Optional. A chip with this name that already runs the model is reused.'),
      el('label', { class: 'mbx-check' }, [spot, ' Spot chip: cheaper, and Google can stop it at any time']),
      el('label', { class: 'mbx-check' }, [agents, ' Coding agents: start the server with the long-context agent settings']),
      el('div', { class: 'app-row' }, [start, stop, reconnect, down]),
      log,
      el('div', { class: 'app-row' }, [el('b', {}, ['Chips in your project']), st]),
      chips,
      note('Quitting MASHBOX closes the tunnel. The chip keeps running and billing until you delete it here or its time runs out.')], 'app-tpu'));
    return box;
  }

  // ------------------------------------------------------------------------------------------------ Connect your agent
  function agents() {
    main.textContent = '';
    Promise.all([get('/api/state'), get('/api/check')]).then(function (r) {
      var s = r[0], p = r[1];
      var srv = (s.servers || []).filter(function (x) { return x.id === s.active; })[0];
      // settings only for a server that answers now (/api/check, which reaches a TPU server only while its tunnel runs)
      // and only when that check is of this very server (the server in use may change between the two answers)
      if (!srv || !p || p.ok !== true || p.url !== srv.url) {
        main.appendChild(panel('Connect your coding agent', [note(srv ? 'The server in use does not answer right now. Start it again in Servers, then come back here.'
          : 'Start a TPU or add a server first, then come back here.'),
          el('a', { class: 'app-btn pri', href: '/p/servers' }, ['Servers'])]));
        return;
      }
      var base = srv.url, model = (p.models && p.models[0] && p.models[0].id) || srv.model || 'MODEL';
      // a server names its own model id, so it goes into a config file only when it is plain text: no quotes,
      // backslashes, spaces or control characters (the copyable files below could otherwise carry extra settings)
      var odd = [];
      if (!plain(model)) { odd.push('model id'); model = 'MODEL'; }
      if (!plain(base)) { odd.push('address'); base = 'http://127.0.0.1:8000'; }
      var ctx = (p.models && p.models[0] && p.models[0].context) || 32768, compact = Math.floor(ctx * 0.8);
      var token = srv.has_key ? 'YOUR_SERVER_KEY' : 'local';
      var claude = JSON.stringify({
        env: { ANTHROPIC_BASE_URL: base, ANTHROPIC_AUTH_TOKEN: token, ANTHROPIC_MODEL: model, ANTHROPIC_SMALL_FAST_MODEL: model,
          ANTHROPIC_DEFAULT_HAIKU_MODEL: model, ANTHROPIC_DEFAULT_SONNET_MODEL: model, ANTHROPIC_DEFAULT_OPUS_MODEL: model,
          CLAUDE_CODE_SUBAGENT_MODEL: model, CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC: '1', DISABLE_TELEMETRY: '1',
          DISABLE_ERROR_REPORTING: '1', DISABLE_AUTOUPDATER: '1', CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS: '1',
          CLAUDE_CODE_MAX_CONTEXT_TOKENS: String(ctx), CLAUDE_CODE_MAX_OUTPUT_TOKENS: '4096' },
        model: model,
        permissions: { deny: ['Agent', 'CronCreate', 'CronDelete', 'CronList', 'EnterWorktree', 'ExitWorktree', 'ListAgents', 'Monitor',
          'NotebookEdit', 'ReportFindings', 'ScheduleWakeup', 'SendMessage', 'Skill', 'TaskStop', 'WebFetch', 'WebSearch', 'Workflow'] }
      }, null, 2);
      // Codex and Pi read a profile of their own (docs/agents/codex.md, pi.md), so the user's own config files stay as they are
      var codex = 'model_provider = "mashbox"\nmodel = ' + toml(model) + '\nmodel_context_window = ' + ctx + '\nmodel_auto_compact_token_limit = ' + compact +
        '\nmodel_reasoning_effort = "none"\n\n[model_providers.mashbox]\nname = "MASHBOX"\nbase_url = ' + toml(base + '/v1') + '\nwire_api = "responses"\n' +
        (srv.has_key ? 'env_key = "MASHBOX_API_KEY"\n' : '');
      var pi = JSON.stringify({ providers: { mashbox: { baseUrl: base + '/v1', api: 'openai-completions', apiKey: token,
        models: [{ id: model, name: model + ' on MASHBOX', contextWindow: ctx, maxTokens: 4096, reasoning: false }] } } }, null, 2);
      var piSet = JSON.stringify({ defaultProvider: 'mashbox', defaultModel: model,
        compaction: { enabled: true, reserveTokens: 6144, keepRecentTokens: 8192 } }, null, 2);
      main.appendChild(el('div', { class: 'app-stack' }, [
        panel('Connect your coding agent', [
          note('Point Claude Code, Codex, Pi or any OpenAI-compatible agent at the server in use. These settings are for this session only; your own agent settings stay as they are.'),
          odd.length ? note('The server sent a ' + odd.join(' and ') + ' with characters a config file cannot take as they are, so the files below use a placeholder. Fill in the real value yourself.') : null,
          el('div', { class: 'app-kv' }, [el('span', {}, ['Server']), el('b', {}, [srv.name]),
            el('span', {}, ['Base URL']), el('code', { class: 'app-mono', id: 'agent-base' }, [base]),
            el('span', {}, ['OpenAI-compatible']), el('code', { class: 'app-mono' }, [base + '/v1']),
            el('span', {}, ['Model id']), el('code', { class: 'app-mono', id: 'agent-model' }, [model]),
            el('span', {}, ['Key']), el('code', { class: 'app-mono' }, [srv.has_key ? 'the key you saved for this server' : 'any text, for example local'])]),
          srv.from === 'tpu' ? note('The address works while MASHBOX is open: it holds the tunnel to your chip.') : null,
          note('Long agent sessions need the server\'s agent settings (a long context and a prompt cache). Tick "Coding agents" when you start a TPU, or see the guide.'),
          link(DOCS + 'agents-on-tpu', 'Agents on TPU guide')]),
        panel('Claude Code', [copyBox('1. Save as ~/.claude/mashbox.json', claude),
          copyBox('2. Run in your project folder', 'claude --settings ~/.claude/mashbox.json --strict-mcp-config')]),
        panel('Codex', [copyBox('1. Save as ~/.codex/mashbox.config.toml', codex),
          copyBox('2. Run in your project folder', (srv.has_key ? 'export MASHBOX_API_KEY=' + token + '\n' : '') + 'codex --profile mashbox')]),
        panel('Pi', [copyBox('1. Save as ~/.pi/mashbox/models.json', pi), copyBox('2. Save as ~/.pi/mashbox/settings.json', piSet),
          copyBox('3. Run in your project folder', 'PI_CODING_AGENT_DIR=~/.pi/mashbox pi')]),
        panel('Any other agent or app', [copyBox('OpenAI-compatible settings', 'base_url = ' + base + '/v1\nmodel = ' + model + '\napi_key = ' + token)])]));
    });
  }

  // ------------------------------------------------------------------------------------------------ Settings
  function settings() {
    main.textContent = '';
    get('/api/state').then(function (s) {
      var curl = el('input', { value: s.catalog_url || '', placeholder: s.catalog_default, spellcheck: 'false', autocomplete: 'off' });
      var msg = note('');
      var save = el('button', { class: 'pri', type: 'button', on: { click: function () {
        post('/api/settings', { catalog_url: curl.value }).then(function () { return get('/api/catalog?refresh=1'); }).then(function (c) {
          var src = c.source || {};
          msg.textContent = (src.from === 'url' ? 'Read ' + c.models.length + ' models from ' + src.url : 'Could not read it (' + (src.error || '') + '); using ' + (src.from === 'cache' ? 'the saved copy' : 'the catalog in this install')) + '.';
        });
      } } }, ['Save and reload']);
      var reset = el('button', { type: 'button', on: { click: function () { curl.value = ''; save.click(); } } }, ['Use the default']);
      var themes = el('div', { class: 'mbx-seg', role: 'group', 'aria-label': 'Theme' }, ['system', 'dark', 'light'].map(function (t) {
        var cur; try { cur = localStorage.getItem('mbx-theme') || 'system'; } catch (e) { cur = 'system'; }
        return el('button', { type: 'button', class: 'sm' + (cur === t ? ' sel' : ''), 'aria-pressed': String(cur === t), on: { click: function (e) {
          if (window.MBX) MBX.theme(t);
          Array.prototype.forEach.call(themes.children, function (b) { b.classList.toggle('sel', b === e.currentTarget); b.setAttribute('aria-pressed', String(b === e.currentTarget)); });
        } } }, [t[0].toUpperCase() + t.slice(1)]);
      }));
      var rtSrc = el('code', { class: 'app-mono' }, ['checking']), rtRel = el('b', {}, ['checking']), drift = note('');
      main.appendChild(el('div', { class: 'app-stack' }, [
        panel('Model catalog', [note('MASHBOX reads its model list when it opens, so new models show up without an app update. The default list is used only when it carries the MASHBOX Labs signature; a saved copy is used when offline.'),
          field('Catalog address', curl, 'Empty uses ' + s.catalog_default), el('div', { class: 'app-row' }, [save, reset]), msg]),
        panel('Look', [themes]),
        panel('About', [el('div', { class: 'app-kv' }, [
          el('span', {}, ['App']), el('b', {}, [s.app_version || 'browser']),
          el('span', {}, ['Runtime']), el('b', {}, ['mashbox ' + s.version]),
          el('span', {}, ['Built from']), rtSrc,
          el('span', {}, ['Latest release']), rtRel,
          el('span', {}, ['Data folder']), el('code', { class: 'app-mono' }, [s.data_dir])]),
          drift,
          note(s.updates ? 'Updates: MASHBOX checks for a new version each time it opens and installs it only when it carries the MASHBOX Labs signature.'
            : 'Updates come with the MASHBOX app; this browser view uses the mashbox you installed.')])]));
      get('/api/release').then(function (r) {
        rtSrc.textContent = r.source || 'not recorded';
        rtRel.textContent = r.latest ? 'mashbox ' + r.latest : 'not reachable';
        drift.textContent = r.newer === true ? 'A newer mashbox is out (' + r.latest + '); this app runs ' + r.version + '.'
          : r.release_file ? 'The runtime is the published release file.' : 'The runtime is a build of mashbox from source, not the published release file.';
      }).catch(function () { rtRel.textContent = 'not reachable'; });
    });
  }

  // ------------------------------------------------------------------------------------------------ World model
  function world() {
    main.textContent = '';
    get('/api/catalog').then(function (c) {
      var rows = c.models.filter(function (m) { return m.module === 'world'; });
      main.appendChild(panel('World model', [note('A world model runs on its own TPU server and opens here when it is running.')]
        .concat(rows.map(function (m) {
          return el('article', { class: 'app-item' }, [el('div', { class: 'app-item-main' }, [el('b', { class: 'app-name' }, [m.name]),
            el('p', {}, [m.summary]), el('div', { class: 'app-tags' }, (m.chips || []).map(function (x) { return el('span', { class: 'mbx-chip' }, [x]); }))]),
            el('div', { class: 'app-acts' }, [el('a', { class: 'app-btn pri', href: '/p/servers?model=' + encodeURIComponent(m.id) }, ['Run on a TPU'])])]);
        }))));
    });
  }

  ({ models: models, servers: servers, agents: agents, settings: settings, world: world }[view] || servers)();
})();
