"""OpenAI-compatible HTTP server on 127.0.0.1 (reach it from your machine with an SSH tunnel).

Routes: GET /health, GET /v1/models, POST /v1/chat/completions (with tools), POST /v1/completions,
POST /v1/messages and /v1/messages/count_tokens (Anthropic Messages, for Claude Code), POST /v1/responses (OpenAI
Responses, for Codex); stream true or false. See agents.py for the coding-agent APIs.
Requests sent together (within GATHER_S of the first) run as one batch of up to S rows (the engine's slot count), so
several users share each decode step. Requests that arrive while a batch is running wait for it to finish, then run
together as the next batch. Send "merge": false to run a request alone.

Who may call it (runtime/guard.py): a request must name the server as localhost, 127.0.0.1, [::1], another IP address
or its own --host (a DNS-rebinding page names its own domain and is refused); a browser request must come from the
server's own origin; a POST must be Content-Type application/json, so a web page cannot send one without a CORS
preflight, which this server never approves. Clients that are not browsers (curl, SDKs, coding agents) pass as before.
With API keys (cli.py: --api-key-file or MASHBOX_API_KEY; made for you when --host is not loopback) every /v1 request
needs one (Authorization: Bearer or x-api-key), and each key's prompts have a prefix cache of their own: a cached prompt
is reused, and reported as cached_tokens, only for requests made with the same key.
"""
import collections
import json
import os
import queue
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import agents as A
from . import guard as G
from .agents import BadRequest, TooLong

GATHER_S = 0.02      # how long the first request of a batch waits for others sent at the same time
KNOBS = dict(max_tokens=512, temperature=0.0, top_p=1.0, top_k=0, min_p=0.0, seed=None, stop=None,
             repetition_penalty=1.0, frequency_penalty=0.0, presence_penalty=0.0)


def knobs_of(b, max_key=None):
    k = {n: b.get(n, d) for n, d in KNOBS.items()}
    if b.get("max_completion_tokens") is not None:
        k["max_tokens"] = b["max_completion_tokens"]
    if max_key is not None:       # agent APIs: their own max key; absent = up to the end of the context (set later)
        k["max_tokens"] = b.get(max_key) if b.get(max_key) is not None else (k["max_tokens"] if b.get("max_tokens") is not None
                                                                            or b.get("max_completion_tokens") is not None else 1 << 30)
    if b.get("stop_sequences") is not None and k["stop"] is None:
        k["stop"] = b["stop_sequences"] or None
    try:
        k["max_tokens"], k["temperature"] = int(k["max_tokens"]), float(k["temperature"] or 0)
        k["top_p"], k["top_k"], k["min_p"] = float(1.0 if k["top_p"] is None else k["top_p"]), int(k["top_k"] or 0), float(k["min_p"] or 0)
        for n_, d_ in (("repetition_penalty", 1.0), ("frequency_penalty", 0.0), ("presence_penalty", 0.0)):
            k[n_] = float(d_ if k[n_] is None else k[n_])
    except (TypeError, ValueError):
        raise BadRequest("max_tokens, temperature, top_p, top_k, min_p and the penalties must be numbers")
    if not (k["max_tokens"] >= 1 and 0 <= k["temperature"] <= 4 and 0 < k["top_p"] <= 1 and k["top_k"] >= 0 and 0 <= k["min_p"] < 1
            and k["repetition_penalty"] > 0 and -2 <= k["frequency_penalty"] <= 2 and -2 <= k["presence_penalty"] <= 2):
        raise BadRequest("max_tokens >= 1, temperature 0..4, top_p (0, 1], top_k >= 0, min_p [0, 1), repetition_penalty > 0, "
                         "frequency_penalty and presence_penalty -2..2")
    if isinstance(k["stop"], str):
        k["stop"] = [k["stop"]]
    if k["stop"] is not None and not (isinstance(k["stop"], list) and all(isinstance(x, str) and x for x in k["stop"])):
        raise BadRequest("stop must be a string or a list of non-empty strings")
    if b.get("n", 1) != 1:
        raise BadRequest("n must be 1")
    if k["seed"] is None:
        k["seed"] = int.from_bytes(os.urandom(7), "big")
    else:
        try:
            k["seed"] = int(k["seed"])
        except (TypeError, ValueError):
            raise BadRequest("seed must be an integer")
    return k


def samp_of(k):
    if k["temperature"] == 0 and k["repetition_penalty"] == 1.0 and k["frequency_penalty"] == 0 and k["presence_penalty"] == 0:
        return None
    return dict(temperature=k["temperature"], top_k=k["top_k"], top_p=k["top_p"], min_p=k["min_p"], seed=k["seed"],
                repetition_penalty=k["repetition_penalty"], frequency_penalty=k["frequency_penalty"],
                presence_penalty=k["presence_penalty"])


class Text:
    """Incremental detokenisation with stop strings: emits only text that can no longer change."""

    def __init__(self, chat, stops):
        self.chat, self.stops, self.sent, self.done = chat, stops or [], 0, False
        self.hold = max((len(s) for s in self.stops), default=1) - 1

    def update(self, ids, final=False):
        text = self.chat.decode(ids)
        cut = min([text.find(s) for s in self.stops if s in text] or [-1]) if self.stops else -1
        if cut >= 0:
            text, final, self.done = text[:cut], True, True
        end = len(text) if final else max(self.sent, len(text) - self.hold - (1 if text.endswith("�") else 0))
        delta = text[self.sent:end]
        self.sent = end
        return delta


class Server:
    def __init__(self, engine, chat, model_id, info, log=None):
        self.engine, self.chat, self.model_id, self.info = engine, chat, model_id, info
        self.log = log or (lambda m: print(m, file=sys.stderr, flush=True))
        im = chat.encode("<|im_start|>")
        self.im_start = im[0] if len(im) == 1 else None
        self.q = queue.Queue()
        self.ready = False
        threading.Thread(target=self.loop, daemon=True).start()

    # -------------------------------------------------------------------------------------------------- batching
    def loop(self):
        hold = collections.deque()
        S = self.engine.S
        while True:
            first = hold.popleft() if hold else self.q.get()
            # Requests sent together reach the queue a few milliseconds apart; wait up to GATHER_S for them, so they share
            # one batch instead of the first one running alone while the rest wait for it to finish.
            end = time.monotonic() + (GATHER_S if first["merge"] else 0.0)
            while len(hold) < S - 1:
                try:
                    hold.append(self.q.get(timeout=max(0.0, end - time.monotonic())))
                except queue.Empty:
                    break
            while True:
                try:
                    hold.append(self.q.get_nowait())
                except queue.Empty:
                    break
            batch = [first]
            if first["merge"]:
                keep = collections.deque()
                for nx in hold:
                    (batch.append(nx) if nx["merge"] and len(batch) < S else keep.append(nx))
                hold = keep
            self.run(batch)

    def run(self, batch):
        mx = max(j["max_new"] for j in batch)

        def tap(ids, fin):
            for i, j in enumerate(batch):
                if "cached" not in j:
                    j["cached"] = (getattr(self.engine, "cached_now", None) or [0] * len(batch))[i]
                if j.get("stream") is not None:
                    j["stream"].put((ids[i][:j["max_new"]], fin[i] is not None or len(ids[i]) >= j["max_new"], fin[i]))
        try:
            kw = dict(marks=[j.get("mark") for j in batch]) if getattr(self.engine, "prefix_cache", False) else {}
            if kw and any(j.get("scope") is not None for j in batch):     # API keys: each key's cache is its own
                kw["scopes"] = [j.get("scope") for j in batch]
            r = self.engine.generate([j["ids"] for j in batch], max_new=mx, stop_ids=self.chat.stop_ids,
                                     samp=[j["samp"] for j in batch], on_step=tap, **kw)
            for i, j in enumerate(batch):
                ids = r["ids"][i]
                j["result"] = dict(ids=ids[:j["max_new"]], finish="length" if len(ids) >= j["max_new"] else (r["finish"][i] or "length"),
                                   batch_rows=len(batch), timing=r["timing"], cached=(r.get("cached") or [0] * len(batch))[i])
        except Exception as e:      # reported to every request of the batch
            for j in batch:
                j["error"] = e
        for j in batch:
            j["ev"].set()
            if j.get("stream") is not None:
                j["stream"].put(None)

    def submit(self, ids, k, merge, stream, mark=None, clamp=False, scope=None):
        room = self.engine.CTX - len(ids)
        if clamp:
            if room < 1:
                raise TooLong(f"prompt is too long: {len(ids)} tokens > {self.engine.CTX - 1} maximum "
                              f"(this server runs a context of {self.engine.CTX} tokens; start it with a larger --ctx or compact)")
            k["max_tokens"] = min(k["max_tokens"], room)
        if len(ids) + k["max_tokens"] > self.engine.CTX:
            raise BadRequest(f"prompt ({len(ids)} tokens) + max_tokens ({k['max_tokens']}) is over the context of {self.engine.CTX}")
        job = dict(ids=ids, max_new=k["max_tokens"], samp=samp_of(k), merge=merge, ev=threading.Event(),
                   stream=queue.Queue() if stream else None, mark=mark, scope=scope)
        self.q.put(job)
        return job

    # -------------------------------------------------------------------------------------------------- routes
    def prompt_ids(self, b, chat):
        if chat:
            ms = b.get("messages")
            if not isinstance(ms, list) or not ms:
                raise BadRequest("messages must be a non-empty list")
            kw = b.get("chat_template_kwargs") or {}
            think = bool(kw.get("enable_thinking", b.get("enable_thinking", False)))
            try:
                return self.chat.chat_ids([dict(role=m.get("role"), content=m.get("content") or "") for m in ms], think)
            except Exception as e:
                raise BadRequest(f"chat template: {e}")
        p = b.get("prompt")
        if isinstance(p, list) and len(p) == 1:
            p = p[0]
        if not isinstance(p, str) or not p:
            raise BadRequest("prompt must be a non-empty string")
        return self.chat.encode(p)

    def models(self):
        return dict(object="list", data=[dict(id=self.model_id, object="model", owned_by="mashbox-labs", created=0,
                                              max_model_len=self.engine.CTX, **self.info)])

    def complete(self, b, chat, write_sse=None, meta=None):
        k = knobs_of(b)
        ids = self.prompt_ids(b, chat)
        job = self.submit(ids, k, bool(b.get("merge", True)), write_sse is not None, scope=(meta or {}).get("scope"))
        rid = ("chatcmpl-" if chat else "cmpl-") + uuid.uuid4().hex[:24]
        created = int(time.time())
        obj = "chat.completion" if chat else "text_completion"
        if write_sse is not None:
            tx = Text(self.chat, k["stop"])
            if chat:
                write_sse(dict(id=rid, object=obj + ".chunk", created=created, model=self.model_id,
                               choices=[dict(index=0, delta=dict(role="assistant"), finish_reason=None)]))
            finish = None
            while True:
                ev = job["stream"].get()
                if ev is None:
                    break
                got, done, fin = ev
                if tx.done:
                    continue
                delta = tx.update(got, final=done)
                if done:
                    finish = "stop" if (fin == "stop" or tx.done) else "length"
                if delta or done:
                    ch = dict(index=0, finish_reason=finish if done else None)
                    if chat:
                        ch["delta"] = dict(content=delta)
                    else:
                        ch["text"] = delta
                    write_sse(dict(id=rid, object=obj + ".chunk" if chat else obj, created=created, model=self.model_id, choices=[ch]))
            job["ev"].wait()
            if "error" in job:
                raise job["error"]
            return None
        job["ev"].wait()
        if "error" in job:
            raise job["error"]
        r = job["result"]
        tx = Text(self.chat, k["stop"])
        text = tx.update(r["ids"], final=True)
        finish = "stop" if (r["finish"] == "stop" or tx.done) else "length"
        n_out = len(r["ids"])
        choice = dict(index=0, finish_reason=finish)
        choice.update(dict(message=dict(role="assistant", content=text)) if chat else dict(text=text))
        return dict(id=rid, object=obj, created=created, model=self.model_id, choices=[choice],
                    usage=dict(prompt_tokens=len(ids), completion_tokens=n_out, total_tokens=len(ids) + n_out),
                    system_fingerprint=f"seed-{k['seed']}")


    # -------------------------------------------------------------------------------------------------- agent APIs
    def mark_of(self, ids):
        """start of the last message block (its <|im_start|>), remembered by the prefix cache next to the prompt end"""
        if self.im_start is None:
            return None
        for j in range(len(ids) - 1, 0, -1):
            if ids[j] == self.im_start:
                return j
        return None

    def run_chat(self, msgs, tools, think, k, merge=True, on_start=None, on_event=None, meta=None):
        """Render, generate, parse. on_start(prompt_tokens) runs once the request is accepted; on_event(ev) gets each
        ("reasoning", s) / ("text", s) / ("tool", name, args). -> dict(text, reasoning, calls, finish, usage)."""
        t0 = time.monotonic()
        try:
            ids = self.chat.chat_ids(msgs, think, tools or None)
        except Exception as e:
            raise BadRequest(f"chat template: {e}")
        job = self.submit(ids, k, merge, True, mark=self.mark_of(ids), clamp=True, scope=(meta or {}).get("scope"))
        if on_start is not None:
            on_start(len(ids))
        par, tx = A.OutParser(tools, think), Text(self.chat, k["stop"])
        out = dict(text=[], reasoning=[], calls=[], finish="length", stop_sequence=None)
        t_first, n_out = None, 0
        while True:
            ev = job["stream"].get()
            if ev is None:
                break
            got, done, fin = ev
            if tx.done:
                continue
            n_out = len(got)
            if t_first is None and got:
                t_first = time.monotonic()
            delta = tx.update(got, final=done)
            end = done or tx.done
            for e in par.feed(delta, final=end):
                (out["calls"].append(e[1:]) if e[0] == "tool" else out[e[0]].append(e[1]))
                if on_event is not None:
                    on_event(e)
            if end:
                out["finish"] = "stop" if (fin == "stop" or tx.done) else "length"
                if tx.done:
                    out["stop_sequence"] = next((s for s in k["stop"] or [] if s in self.chat.decode(got)), None)
        job["ev"].wait()
        if "error" in job:
            raise job["error"]
        cached = job.get("cached", job.get("result", {}).get("cached", 0))
        out["usage"] = dict(prompt=len(ids), completion=n_out, cached=cached)
        self.log(json.dumps(dict(event="request", api=(meta or {}).get("api"), ua=(meta or {}).get("ua"), prompt_tokens=len(ids),
                                 cached_tokens=cached, new_tokens=n_out, tools=len(tools or []), tool_calls=len(out["calls"]),
                                 thinking=bool(think), finish=out["finish"], dropped_tools=(meta or {}).get("dropped_tools"), batch_rows=job.get("result", {}).get("batch_rows"),
                                 ttft_s=round((t_first or time.monotonic()) - t0, 3), total_s=round(time.monotonic() - t0, 3))))
        return out

    def count_tokens(self, b):
        cv, tools, extra = A.from_anthropic(b)
        think = (b.get("thinking") or {}).get("type") in ("enabled", "adaptive")
        try:
            return dict(input_tokens=len(self.chat.chat_ids(cv.messages(extra), think, tools or None)))
        except Exception as e:
            raise BadRequest(f"chat template: {e}")

    def openai_chat(self, b, sse=None, meta=None):
        k = knobs_of(b, "max_completion_tokens")
        cv, tools, extra = A.from_openai_chat(b)
        kw = b.get("chat_template_kwargs") or {}
        think = bool(kw.get("enable_thinking", b.get("enable_thinking", False)))
        cid, created = A.rid("chatcmpl-"), int(time.time())
        head = dict(id=cid, object="chat.completion.chunk", created=created, model=self.model_id)
        n_calls = [0]

        def start(_n):
            sse.start()
            sse.data(dict(head, choices=[dict(index=0, delta=dict(role="assistant", content=""), finish_reason=None)]))

        def event(e):
            if e[0] == "text":
                d = dict(content=e[1])
            elif e[0] == "reasoning":
                d = dict(reasoning_content=e[1])
            else:
                d = dict(tool_calls=[dict(index=n_calls[0], id=A.rid("call_"), type="function",
                                          function=dict(name=e[1], arguments=json.dumps(e[2], ensure_ascii=False)))])
                n_calls[0] += 1
            sse.data(dict(head, choices=[dict(index=0, delta=d, finish_reason=None)]))
        r = self.run_chat(cv.messages(extra), tools, think, k, bool(b.get("merge", True)),
                          start if sse else None, event if sse else None, meta)
        fin = "tool_calls" if r["calls"] else r["finish"]
        u = r["usage"]
        usage = dict(prompt_tokens=u["prompt"], completion_tokens=u["completion"], total_tokens=u["prompt"] + u["completion"],
                     prompt_tokens_details=dict(cached_tokens=u["cached"]))
        if sse:
            sse.data(dict(head, choices=[dict(index=0, delta={}, finish_reason=fin)]))
            if (b.get("stream_options") or {}).get("include_usage"):
                sse.data(dict(head, choices=[], usage=usage))
            sse.done()
            return None
        msg = dict(role="assistant", content="".join(r["text"]) or (None if r["calls"] else ""))
        if r["reasoning"]:
            msg["reasoning_content"] = "".join(r["reasoning"])
        if r["calls"]:
            msg["tool_calls"] = [dict(id=A.rid("call_"), type="function", function=dict(name=n, arguments=json.dumps(a, ensure_ascii=False)))
                                 for n, a in r["calls"]]
        return dict(id=cid, object="chat.completion", created=created, model=self.model_id,
                    choices=[dict(index=0, message=msg, finish_reason=fin)], usage=usage)

    def anthropic(self, b, sse=None, meta=None):
        k = knobs_of(b, "max_tokens")
        cv, tools, extra = A.from_anthropic(b)
        think = (b.get("thinking") or {}).get("type") in ("enabled", "adaptive")
        mid = A.rid("msg_")
        blocks, cur = [], [None]          # finished content blocks, open block type

        def start(n):
            sse.start()
            sse.event("message_start", dict(type="message_start", message=dict(
                id=mid, type="message", role="assistant", model=b.get("model") or self.model_id, content=[], stop_reason=None,
                stop_sequence=None, usage=dict(input_tokens=n, output_tokens=0, cache_read_input_tokens=0, cache_creation_input_tokens=0))))
            sse.event("ping", dict(type="ping"))

        def close():
            if cur[0] is None:
                return
            if cur[0] == "thinking":
                sse.event("content_block_delta", dict(type="content_block_delta", index=len(blocks) - 1,
                                                      delta=dict(type="signature_delta", signature="mashbox-local")))
            sse.event("content_block_stop", dict(type="content_block_stop", index=len(blocks) - 1))
            cur[0] = None

        def event(e):
            kind = dict(text="text", reasoning="thinking", tool="tool_use")[e[0]]
            if kind == "tool_use" or cur[0] != kind:
                close()
                cb = (dict(type="text", text="") if kind == "text" else dict(type="thinking", thinking="") if kind == "thinking"
                      else dict(type="tool_use", id=A.rid("toolu_"), name=e[1], input={}))
                blocks.append(cb)
                sse.event("content_block_start", dict(type="content_block_start", index=len(blocks) - 1, content_block=cb))
                cur[0] = kind
            d = (dict(type="text_delta", text=e[1]) if kind == "text" else dict(type="thinking_delta", thinking=e[1]) if kind == "thinking"
                 else dict(type="input_json_delta", partial_json=json.dumps(e[2], ensure_ascii=False)))
            sse.event("content_block_delta", dict(type="content_block_delta", index=len(blocks) - 1, delta=d))
            if kind == "tool_use":
                close()
        r = self.run_chat(cv.messages(extra), tools, think, k, True, start if sse else None, event if sse else None, meta)
        stop = "tool_use" if r["calls"] else ("max_tokens" if r["finish"] == "length" else
                                               "stop_sequence" if r["stop_sequence"] else "end_turn")
        u = r["usage"]
        usage = dict(input_tokens=u["prompt"] - u["cached"], cache_read_input_tokens=u["cached"], cache_creation_input_tokens=0,
                     output_tokens=u["completion"])
        if sse:
            close()
            sse.event("message_delta", dict(type="message_delta", delta=dict(stop_reason=stop, stop_sequence=r["stop_sequence"]),
                                            usage=usage))
            sse.event("message_stop", dict(type="message_stop"))
            return None
        content = []
        if r["reasoning"]:
            content.append(dict(type="thinking", thinking="".join(r["reasoning"]), signature="mashbox-local"))
        if r["text"]:
            content.append(dict(type="text", text="".join(r["text"])))
        content += [dict(type="tool_use", id=A.rid("toolu_"), name=n, input=a) for n, a in r["calls"]]
        return dict(id=mid, type="message", role="assistant", model=b.get("model") or self.model_id, content=content,
                    stop_reason=stop, stop_sequence=r["stop_sequence"], usage=usage)

    def responses(self, b, sse=None, meta=None):
        k = knobs_of(b, "max_output_tokens")
        cv, tools, extra, kinds, dropped = A.from_responses(b)
        if dropped:
            meta = dict(meta or {}, dropped_tools=dropped)
        eff = (b.get("reasoning") or {}).get("effort")
        think = eff not in (None, "none", "minimal")
        resp = dict(id=A.rid("resp_"), object="response", created_at=int(time.time()), status="in_progress",
                    model=b.get("model") or self.model_id, output=[], parallel_tool_calls=bool(b.get("parallel_tool_calls", False)),
                    tool_choice=b.get("tool_choice", "auto"), tools=b.get("tools") or [], usage=None)
        seq, items, cur = [0], [], [None]

        def ev(name, **o):
            seq[0] += 1
            sse.event(name, dict(type=name, sequence_number=seq[0], **o))

        def start(_n):
            sse.start()
            ev("response.created", response=resp)
            ev("response.in_progress", response=resp)

        def close():
            it = cur[0]
            if it is None:
                return
            oi = len(items) - 1
            if it["type"] == "message":
                part = dict(type="output_text", text=it["_t"], annotations=[])
                ev("response.output_text.done", item_id=it["id"], output_index=oi, content_index=0, text=it["_t"])
                ev("response.content_part.done", item_id=it["id"], output_index=oi, content_index=0, part=part)
                it["content"], it["status"] = [part], "completed"
            elif it["type"] == "reasoning":
                ev("response.reasoning_text.done", item_id=it["id"], output_index=oi, content_index=0, text=it["_t"])
                it["content"] = [dict(type="reasoning_text", text=it["_t"])]
            it.pop("_t", None)
            ev("response.output_item.done", output_index=oi, item=it)
            cur[0] = None

        def event(e):
            kind = dict(text="message", reasoning="reasoning", tool="call")[e[0]]
            if kind == "call":
                close()
                n, a = e[1], e[2]
                kind_, ns = kinds.get(n, ("function", None))
                if kind_ == "custom":
                    it = dict(type="custom_tool_call", id=A.rid("ctc_"), call_id=A.rid("call_"), name=n,
                              input=a.get("input", "") if isinstance(a.get("input", ""), str) else json.dumps(a.get("input")),
                              status="completed")
                else:
                    it = dict(type="function_call", id=A.rid("fc_"), call_id=A.rid("call_"), name=n,
                              arguments=json.dumps(a, ensure_ascii=False), status="completed")
                    if ns:
                        it["namespace"] = ns
                items.append(it)
                if sse:
                    ev("response.output_item.added", output_index=len(items) - 1, item=dict(it, status="in_progress"))
                    if it["type"] == "function_call":
                        ev("response.function_call_arguments.delta", item_id=it["id"], output_index=len(items) - 1, delta=it["arguments"])
                        ev("response.function_call_arguments.done", item_id=it["id"], output_index=len(items) - 1,
                           arguments=it["arguments"])
                    else:
                        ev("response.custom_tool_call_input.delta", item_id=it["id"], output_index=len(items) - 1, delta=it["input"])
                        ev("response.custom_tool_call_input.done", item_id=it["id"], output_index=len(items) - 1, input=it["input"])
                    ev("response.output_item.done", output_index=len(items) - 1, item=it)
                return
            if cur[0] is None or cur[0]["type"] != kind:
                close()
                it = (dict(type="message", id=A.rid("msg_"), status="in_progress", role="assistant", content=[], _t="")
                      if kind == "message" else dict(type="reasoning", id=A.rid("rs_"), summary=[], _t=""))
                items.append(it)
                cur[0] = it
                if sse:
                    ev("response.output_item.added", output_index=len(items) - 1, item={x: y for x, y in it.items() if x != "_t"})
                    if kind == "message":
                        ev("response.content_part.added", item_id=it["id"], output_index=len(items) - 1, content_index=0,
                           part=dict(type="output_text", text="", annotations=[]))
            it = cur[0]
            it["_t"] += e[1]
            if sse:
                ev("response.output_text.delta" if kind == "message" else "response.reasoning_text.delta", item_id=it["id"],
                   output_index=len(items) - 1, content_index=0, delta=e[1])

        if sse is None:
            class _Quiet:      # non-streaming: build the same items, write nothing
                def event(self, *a):
                    pass
            sse = _Quiet()
        r = self.run_chat(cv.messages(extra), tools, think, k, True, start if hasattr(sse, "start") else None, event, meta)
        close()
        u = r["usage"]
        resp.update(output=items, usage=dict(input_tokens=u["prompt"], input_tokens_details=dict(cached_tokens=u["cached"]),
                                             output_tokens=u["completion"], output_tokens_details=dict(reasoning_tokens=0),
                                             total_tokens=u["prompt"] + u["completion"]))
        if r["finish"] == "length" and not r["calls"]:
            resp.update(status="incomplete", incomplete_details=dict(reason="max_output_tokens"))
        else:
            resp["status"] = "completed"
        if hasattr(sse, "start"):
            ev("response.incomplete" if resp["status"] == "incomplete" else "response.completed", response=resp)
            return None
        return resp


class SSE:
    """Server-sent events on one response: data-only lines (OpenAI chat) or named events (Anthropic, Responses)."""

    def __init__(self, h):
        self.h, self.started = h, False

    def start(self):
        h = self.h
        h.send_response(200)
        h.send_header("Content-Type", "text/event-stream")
        h.send_header("Cache-Control", "no-cache")
        h.send_header("Connection", "close")
        h.end_headers()
        h.close_connection = True
        self.started = True

    def _w(self, b):
        self.h.wfile.write(b)
        self.h.wfile.flush()

    def data(self, o):
        self._w(b"data: " + json.dumps(o, ensure_ascii=False).encode() + b"\n\n")

    def event(self, name, o):
        self._w(b"event: " + name.encode() + b"\ndata: " + json.dumps(o, ensure_ascii=False).encode() + b"\n\n")

    def done(self):
        self._w(b"data: [DONE]\n\n")

    def error(self, msg, anth, typ="invalid_request_error"):
        if anth:
            self.event("error", dict(type="error", error=dict(type=typ, message=msg)))
        else:
            self.data(dict(error=dict(message=msg, type=typ)))
            self.done()


def key_scope(headers, keys):
    """The index of the API key a request carries (Authorization: Bearer or x-api-key), None when it carries none of them.
    Every key is compared in full, in constant time."""
    a = headers.get("Authorization") or ""
    given = [g for g in (a[7:].strip() if a[:7].lower() == "bearer " else None, headers.get("x-api-key")) if g]
    hit = None
    for i, k in enumerate(keys):
        for g in given:
            if G._same(g, k) and hit is None:
                hit = i
    return hit


def serve(server, host="127.0.0.1", port=8000, log=print, keys=None):
    """keys: API keys a request must carry on /v1 (None or empty: no key). Each key is its own prefix cache scope.
    /health stays open, as llama-server's public endpoints."""
    S = server
    keys = [k for k in (keys or []) if k]
    guard = G.Guard(hosts=[host])

    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        scope = None

        def _gate(self, key=True):
            """Host, Origin and Content-Type for every request, and the API key when `key`. True = answered (refused)."""
            err = guard.check(self.command, self.path, self.headers, auth=False)
            if err is None and keys and key:
                self.scope = key_scope(self.headers, keys)
                if self.scope is None:
                    err = (401, "missing or wrong API key: send Authorization: Bearer <key> or x-api-key: <key>")
            if err is None:
                return False
            anth = self.path.startswith("/v1/messages")
            typ = {401: "authentication_error", 403: "permission_error"}.get(err[0], "invalid_request_error")
            self._err(err[0], err[1], typ, anth=anth)
            self.close_connection = True
            return True

        def _send(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _err(self, code, msg, typ="invalid_request_error", anth=False):
            if anth:     # Anthropic error shape
                typ = {"invalid_request_error": "invalid_request_error", "unavailable": "overloaded_error"}.get(typ, typ)
                return self._send(code, dict(type="error", error=dict(type=typ, message=msg)))
            self._send(code, dict(error=dict(message=msg, type=typ)))

        def do_HEAD(self):
            if self._gate(key=False):
                return
            self.send_response(200 if S.ready else 503)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_GET(self):
            path = self.path.split("?")[0].rstrip("/") or "/"
            if self._gate(key=path not in ("/health", "/v1/health", "/")):      # /health needs no key
                return
            if path in ("/health", "/v1/health", "/"):
                return self._send(200 if S.ready else 503, dict(status="ok" if S.ready else "loading"))
            if path == "/v1/models":
                return self._send(200, S.models())
            if path.startswith("/v1/models/"):
                m = S.models()["data"][0]
                return self._send(200, m) if path.endswith("/" + m["id"]) else self._err(404, f"model {path.rsplit('/', 1)[1]} not found")
            self._err(404, "not found")

        def do_POST(self):
            path = self.path.split("?")[0].rstrip("/")
            api = {"/v1/chat/completions": "chat", "/v1/completions": "completions", "/v1/messages": "anthropic",
                   "/v1/messages/count_tokens": "count_tokens", "/v1/responses": "responses"}.get(path)
            anth = api in ("anthropic", "count_tokens")
            if self._gate():
                return
            if api is None:
                return self._err(404, f"not found: POST {path}", anth=anth)
            try:
                n = int(self.headers.get("Content-Length") or 0)
                b = json.loads(self.rfile.read(n) or b"{}")
                if not isinstance(b, dict):
                    raise BadRequest("body must be a JSON object")
            except (ValueError, BadRequest) as e:
                return self._err(400, f"bad JSON body: {e}", anth=anth)
            if not S.ready:
                return self._err(503, "model is loading", "unavailable" if not anth else "overloaded_error", anth=anth)
            meta = dict(api=api, ua=(self.headers.get("User-Agent") or "")[:80], scope=self.scope)
            if os.environ.get("MASHBOX_DUMP_DIR"):        # debugging client compatibility: keep each request body
                try:
                    with open(os.path.join(os.environ["MASHBOX_DUMP_DIR"], f"{time.time():.3f}-{api}.json"), "w") as f:
                        json.dump(dict(path=self.path, headers={k: v for k, v in self.headers.items()
                                                                if k.lower() not in ("authorization", "x-api-key")}, body=b), f)
                except OSError:
                    pass
            sse = SSE(self) if b.get("stream") else None
            try:
                if api == "count_tokens":
                    return self._send(200, S.count_tokens(b))
                if api == "completions":
                    return self.legacy(b, False, meta)
                fn = dict(chat=S.openai_chat, anthropic=S.anthropic, responses=S.responses)[api]
                r = fn(b, sse, meta)
                if sse is None:
                    return self._send(200, r)
            except BadRequest as e:
                S.log(json.dumps(dict(event="rejected", api=api, ua=meta["ua"], error=str(e)[:300])))
                if sse is not None and sse.started:
                    return sse.error(str(e), anth)
                return self._err(400, str(e), anth=anth)
            except (BrokenPipeError, ConnectionResetError):
                return
            except Exception as e:
                S.log(json.dumps(dict(event="error", api=api, error=f"{type(e).__name__}: {e}"[:300])))
                if sse is not None and sse.started:
                    return sse.error(f"{type(e).__name__}: {e}", anth, "server_error" if not anth else "api_error")
                return self._err(500, f"{type(e).__name__}: {e}", "server_error" if not anth else "api_error", anth=anth)

        def legacy(self, b, chat, meta=None):
            if b.get("stream"):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.end_headers()
                self.close_connection = True

                def write(o):
                    self.wfile.write(b"data: " + json.dumps(o).encode() + b"\n\n")
                    self.wfile.flush()
                try:
                    S.complete(b, chat, write, meta)
                except BadRequest as e:
                    write(dict(error=dict(message=str(e), type="invalid_request_error")))
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
                return
            return self._send(200, S.complete(b, chat, meta=meta))

        def log_message(self, *a):
            pass

    httpd = ThreadingHTTPServer((host, port), H)
    log(json.dumps(dict(event="listening", url=f"http://{host}:{port}/v1")))
    httpd.serve_forever()
