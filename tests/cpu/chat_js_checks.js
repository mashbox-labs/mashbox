// The chat module's pure helpers (src/mashbox/ui/chat/chat.js) in Node: Markdown, the stream reader, the request
// body, reply timing, export. Run by tests/cpu/test_chat_module.py; prints "ok N" or throws.
'use strict';
const assert = require('assert');
const X = require(process.argv[2]);
let n = 0;
function eq(a, b, m) { assert.deepStrictEqual(a, b, m); n++; }
function has(s, sub, m) { assert.ok(s.includes(sub), (m || '') + '\n--- missing: ' + sub + '\n--- in: ' + s); n++; }
function not(s, sub, m) { assert.ok(!s.includes(sub), (m || '') + '\n--- unexpected: ' + sub + '\n--- in: ' + s); n++; }

// Markdown: model text never becomes markup
const evil = X.md('<img src=x onerror=alert(1)> <script>alert(1)</script> [a](javascript:alert(1)) [b](data:text/html,x) "q" \'s\'');
not(evil, '<img'); not(evil, '<script'); not(evil, 'href="javascript'); not(evil, 'href="data');
has(evil, '&lt;img src=x onerror=alert(1)&gt;');
not(X.md('[x](https://a.b/" onmouseover="alert(1))'), '" onmouseover', 'quotes in a link stay inside the attribute');
has(X.md('`<b>` and **bold** and *it* and ~~gone~~'), '<code>&lt;b&gt;</code>');
has(X.md('**bold** _it_'), '<strong>bold</strong>'); has(X.md('a *it* b'), '<em>it</em>'); has(X.md('~~x~~'), '<del>x</del>');
not(X.md('snake_case_name and 2*3*4'), '<em>', 'no emphasis inside words or arithmetic');
// code blocks, including one still streaming (no closing fence yet)
const cb = X.md('Here:\n\n```js\nconst a = 1 < 2;\n```\nafter');
has(cb, '<pre><code data-lang="js">const a = 1 &lt; 2;</code></pre>'); has(cb, 'data-copy'); has(cb, '<p>after</p>');
has(X.md('```py\nprint(1)\nprint(2'), 'print(1)\nprint(2</code>', 'an open fence renders as code while it streams');
has(X.md('````\n```\ninner\n```\n````'), '```\ninner\n```', 'a longer fence holds a shorter one');
// blocks
has(X.md('# T\n## U'), '<h1>T</h1><h2>U</h2>');
eq(X.md('- a\n- b\n  - c\n- d'), '<ul><li>a</li><li>b<ul><li>c</li></ul></li><li>d</li></ul>');
has(X.md('3. x\n4. y'), '<ol start="3"><li>x</li><li>y</li></ol>');
has(X.md('- [x] done\n- [ ] todo'), '<input type="checkbox" disabled checked>');
has(X.md('| a | b |\n|:-:|--:|\n| 1 | 2 |\n| 3 |'), '<th style="text-align:center">a</th>');
has(X.md('| a | b |\n|---|---|\n| 1 | 2 |\n| 3 |'), '<td>3</td><td></td>', 'short rows are padded');
has(X.md('> quoted\n> more'), '<blockquote><p>quoted<br>more</p></blockquote>');
eq(X.md('---'), '<hr>');
has(X.md('line one\nline two'), 'line one<br>line two');
has(X.md('see https://mashbox.gg/docs.'), '<a href="https://mashbox.gg/docs" target="_blank" rel="noopener noreferrer">https://mashbox.gg/docs</a>.');
eq(X.md(''), ''); eq(X.md(null), '');

// the stream reader: events split anywhere across chunks
const rd = X.sseReader();
eq(rd('data: {"a":1}\n\ndata: {"b"'), ['{"a":1}']);
eq(rd(':2}\r\n\r\n: comment\n\ndata: [DONE]'), ['{"b":2}']);
eq(rd('', true), ['[DONE]']);

// the request: every engine setting, nothing for an empty one
const caps = { thinking: true };
const s = { temperature: 0.7, top_p: 0.8, top_k: 20, min_p: 0, repetition_penalty: 1, presence_penalty: 1.5, frequency_penalty: -0.5,
  max_tokens: '256', seed: 42, stop: 'END\n\\n\\nUser:', thinking: true };
const b = X.buildBody('m', [{ role: 'user', content: 'hi' }], s, caps);
eq(b, { model: 'm', messages: [{ role: 'user', content: 'hi' }], stream: true, stream_options: { include_usage: true }, temperature: 0.7, top_p: 0.8,
  top_k: 20, min_p: 0, repetition_penalty: 1, presence_penalty: 1.5, frequency_penalty: -0.5, max_tokens: 256, seed: 42,
  stop: ['END', '\n\nUser:'], chat_template_kwargs: { enable_thinking: true } });
const e = X.buildBody('m', [], { temperature: null, top_p: '', max_tokens: null, seed: null, stop: '', thinking: true }, {});
eq(Object.keys(e).sort(), ['messages', 'model', 'stream', 'stream_options'], 'server defaults: no knobs, no thinking key for a model without it');
eq(X.buildBody('m', [], { thinking: false }, caps).chat_template_kwargs, { enable_thinking: false });

// the conversation sent back: the chosen version of each reply, errors left out, thinking kept only when asked
const chat = { system: 'Be brief.', messages: [
  { role: 'user', content: 'q1' },
  { role: 'assistant', sel: 1, versions: [{ content: 'old' }, { content: 'new', reasoning: 'r' }] },
  { role: 'user', content: 'q2' },
  { role: 'assistant', sel: 0, versions: [{ content: '', error: 'boom' }] }] };
eq(X.toMessages(chat), [{ role: 'system', content: 'Be brief.' }, { role: 'user', content: 'q1' }, { role: 'assistant', content: 'new' }, { role: 'user', content: 'q2' }]);
eq(X.toMessages(chat, { keepThinking: true })[2], { role: 'assistant', content: 'new', reasoning_content: 'r' });

// timing: tokens after the first over the time after the first
const st = X.replyStats({ t0: 1000, tFirst: 1500, tEnd: 3500, usage: { prompt_tokens: 40, completion_tokens: 81, prompt_tokens_details: { cached_tokens: 32 } }, finish: 'stop' });
eq([st.ttft, st.tps, st.tokens, st.prompt, st.cached, st.total, st.finish], [0.5, 40, 81, 40, 32, 2.5, 'stop']);
eq(X.replyStats({ t0: 0, tFirst: 100, tEnd: 900, usage: null }).tps, null, 'no usage (a stopped reply): no speed claimed');

// export
const mdx = X.chatMarkdown({ title: 'T', model: 'm', system: 'S', messages: chat.messages.slice(0, 2) });
has(mdx, '# T'); has(mdx, '## You\n\nq1'); has(mdx, '<summary>Thinking</summary>'); has(mdx, 'new');
console.log('ok ' + n);
