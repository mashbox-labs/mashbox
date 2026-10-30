import numpy as np
import pytest
from mashbox.models.hy_nomas import buckets as B
from mashbox.models.hy_nomas import chip


def visible(gq, gk):
    return (gk < 0) | (gk == gq)


def test_bucket_choice():
    assert B.bucket(1) == 256 and B.bucket(256) == 256 and B.bucket(257) == 512 and B.bucket(512) == 512
    with pytest.raises(ValueError):
        B.bucket(513)


@pytest.mark.parametrize("actions", [False, True])
def test_padding_and_head_masks_match_compacted_keys(actions):
    FR, TT, n_text, n_prev = 4, 4, 37, 29
    n_img, hist = FR * TT, 2 * FR
    offs = [15, 26, 37] if actions else None
    fa = [0, 0, 1, 1] if actions else None
    keep = 10 if actions else n_prev
    tb, pb = B.bucket(n_text), B.bucket(n_prev)
    LQ, LK, nv = B.layout(n_img, tb, hist, pb, block_q=16, block_kv=16)
    gq, gk = B.groups(n_img, FR, fa, n_text, offs, tb, LQ, LK, hist, n_prev, keep, pb)
    vis = visible(gq[:n_img + n_text], gk[:, :nv])
    # the same rows over the compacted keys [image | text | history image | previous text kept]
    tg = B.text_groups(n_text, offs)
    gqc = np.concatenate([np.repeat(fa, FR) if actions else np.full(n_img, -1), tg])[:, None]
    gkc = np.concatenate([np.full(n_img, -1), tg, np.full(hist + keep, -1)])[None]
    visc = visible(gqc, gkc)
    idx = np.concatenate([np.arange(n_img + n_text), n_img + tb + np.arange(hist), n_img + tb + hist + np.arange(keep)])
    assert (vis[:, idx] == visc).all()
    assert vis.sum() == visc.sum()          # nothing else is visible: padding and cut text are masked


def test_shape_count_is_fixed():
    lengths = list(range(1, 513))
    assert B.shape_count(lengths) == (2, 3)


def test_chip_label_from_devices():
    class D:
        device_kind = "TPU v6 lite"
    assert chip.chip_label([D()] * 4, zone_name="asia-northeast1-b") == "TPU v6e-4 (4 chips), asia-northeast1-b"
    assert chip.chip_label([D()], zone_name="") == "TPU v6e-1 (1 chip)"



def _viewer_on_link(mbit, rtt, secs=60.0, acks=True, bunch=0.0):
    """One app viewer (server.Client) on a simulated link: a 24 fps stream (110 kB key frame every 48 frames, 30 kB
    otherwise, about 5.8 Mbit/s, the sizes measured on the live stream) leaves through one bottleneck of `mbit` Mbit/s with
    `rtt` s round trip; the viewer acks each frame it receives, as page.html does (or never, as an older page). With
    `bunch` > 0 frames reach the viewer only at multiples of `bunch` s (a lost packet holds up the ones behind it), so
    their acks come back bunched.
    Returns (frames offered, [(offer time, arrival time)] of the frames that went out)."""
    from mashbox.models.hy_nomas import server as SV

    class Clock:
        t = 1000.0

    class Viewer(SV.Client):
        clock = staticmethod(lambda: Clock.t)

    c = Viewer(sock=None, mode='h264')
    rate, free, due, out = mbit * 1e6 / 8, Clock.t, [], []
    n = int(secs * 24)
    for i in range(n):
        t = 1000.0 + i / 24
        while due and due[0][0] <= t:          # acks reach the server in order
            Clock.t, g = due.pop(0)
            c.on_ack(g)
        Clock.t = t
        c.offer(bytes(110000 if i % 48 == 0 else 30000), i % 48 == 0, i + 1)
        if c.vq:                               # it went out: one queue (socket + link) in front of the bottleneck
            m = c.vq.popleft()
            free = max(free, t) + len(m) / rate
            arrive = free + rtt / 2
            if bunch:
                arrive = (int(arrive / bunch) + 1) * bunch
            out.append((t, arrive))
            if acks:
                due.append((arrive + rtt / 2, i + 1))
    return n, out

def lag_p95(got, after=10.0):
    lags = sorted(a - t for t, a in got if t >= 1000.0 + after)
    return lags[int(len(lags) * 0.95)]


def test_viewer_video_stays_live_on_slow_links():
    """The app server keeps old video out of slow links (newest frame wins) and changes nothing on a fast one."""
    n, got = _viewer_on_link(100, 0.002)
    assert len(got) == n and lag_p95(got) < 0.05
    n, got = _viewer_on_link(5, 0.15)
    assert len(got) > n * 0.5 and lag_p95(got) < 1.0
    n, got = _viewer_on_link(1, 0.25)
    assert len(got) > n * 0.05 and lag_p95(got) < 2.0
    n, got = _viewer_on_link(1, 0.25, bunch=0.5)
    assert len(got) > n * 0.05 and lag_p95(got) < 2.5
    n, got = _viewer_on_link(5, 0.15, bunch=0.3)
    assert len(got) > n * 0.2 and lag_p95(got) < 1.5
    n, got = _viewer_on_link(100, 0.002, acks=False)  # an older page that never acks still gets every frame after 5 s
    assert len([1 for t, a in got if t >= 1006.0]) == n - 6 * 24


def test_app_server_answers_only_its_route_table():
    """The Hy-Nomas app server's handlers compare the request path only with the paths in server.ROUTES (GET in do_GET,
    POST in api), so a new route is added to that one table."""
    import ast
    import pathlib
    from mashbox.models.hy_nomas import server
    tree = ast.parse(pathlib.Path(server.__file__).read_text())
    seen = {"GET": set(), "POST": set()}
    for fn in ast.walk(tree):
        if isinstance(fn, ast.FunctionDef) and fn.name in ("do_GET", "do_POST", "api"):
            for n in ast.walk(fn):
                if isinstance(n, ast.Compare) and isinstance(n.left, ast.Name) and n.left.id == "path":
                    for c in ast.walk(n):
                        if isinstance(c, ast.Constant) and isinstance(c.value, str):
                            seen["GET" if fn.name == "do_GET" else "POST"].add(c.value)
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name) \
                        and n.func.value.id == "path" and n.func.attr != "split":   # path.startswith(...): a route as well
                    for a in n.args:   # the one allowed prefix is the shell's files: '/' + shell.ROUTE
                        if not (isinstance(a, ast.BinOp) and isinstance(a.right, ast.Attribute) and a.right.attr == "ROUTE"):
                            seen["GET" if fn.name == "do_GET" else "POST"].add(ast.unparse(a))
    assert seen == {k: set(v) for k, v in server.ROUTES.items()}, seen


def test_kernel_download_error_stops_with_one_plain_line(monkeypatch):
    """A kernel bundle download that the Hub refuses (a private or missing repo answers 401) or that cannot reach the Hub
    stops the run with one line naming the repo and the way out (--no-kernels), not a traceback. The line is hub.fetch's
    (tests/cpu/test_hub_fetch.py, for every download); the Model reaches it through hub.kernel_bundle."""
    requests = pytest.importorskip("requests")
    errors = pytest.importorskip("huggingface_hub.errors")
    import inspect
    import huggingface_hub
    from mashbox import hub, kernels as K
    from mashbox.models.hy_nomas import app
    m = hub.model_entry('hy-nomas')
    row = m['kernels'][0]
    monkeypatch.setattr(K, 'chip_family', lambda: row['chips'][0])
    monkeypatch.setattr(K, 'versions', lambda: dict(jax=row['jax'], jaxlib=None, libtpu=None))
    r = requests.Response()
    r.status_code = 401
    for err in (errors.RepositoryNotFoundError('401 Client Error. Repository Not Found', response=r),
                requests.exceptions.ConnectionError('Max retries exceeded with url')):
        def refuse(*a, err=err, **k):
            raise err
        monkeypatch.setattr(huggingface_hub, 'hf_hub_download', refuse)
        with pytest.raises(hub.DownloadError) as e:
            hub.kernel_bundle(m, print)
        msg = str(e.value)
        assert 'https://huggingface.co/mashbox-labs/Hy-Nomas-TPU' in msg and '--no-kernels' in msg
        assert ('HTTP 401' in msg) == (err.__class__.__name__ == 'RepositoryNotFoundError')
        assert '\n' not in msg and 'Traceback' not in msg
    assert 'hub.kernel_bundle(m, log)' in inspect.getsource(app.Model)


def hy_nomas_extra():
    """the distributions pyproject's hy-nomas extra adds, in its order"""
    import pathlib
    import re
    import tomllib
    proj = tomllib.loads((pathlib.Path(__file__).resolve().parents[2] / "pyproject.toml").read_text())["project"]
    return [re.match(r"[A-Za-z0-9._-]+", r).group(0) for r in proj["optional-dependencies"]["hy-nomas"]]


@pytest.mark.parametrize("uv", [True, False], ids=["uv tool", "pip"])
def test_run_without_the_hy_nomas_extra_stops_before_any_download(monkeypatch, capsys, uv):
    """On a Cloud TPU VM installed with mashbox[tpu] only (install.sh's TPU default), `run hy-nomas` passed the chip check
    and the downloads, then ended in ModuleNotFoundError: torch (upstream.TencentText). It stops first: one line naming
    the distributions the hy-nomas extra adds and the line that installs them (install.sh with MASHBOX_EXTRAS in a uv tool,
    else the models.json install line), exit 2; nothing is fetched and no chip is opened. --stub (no model) and download
    (files only) still run without them."""
    from mashbox import cli, hub
    from mashbox.models.hy_nomas import app, upstream, weights
    missing = hy_nomas_extra()
    monkeypatch.setattr(cli, "installed", lambda d: d not in missing, raising=False)
    monkeypatch.setattr(cli, "uv_tool", lambda: uv, raising=False)

    def ran(*a, **k):
        raise AssertionError("reached the chip check or a download")
    monkeypatch.setattr(hub, "fetch", ran)
    monkeypatch.setattr(app, "run", ran)                   # Model(a): the chip check, then every download
    assert cli.main(["run", "hy-nomas", "--ui"]) == 2
    err = capsys.readouterr().err.strip().splitlines()
    fix = ("curl -fsSL https://mashbox.gg/install.sh | MASHBOX_EXTRAS=tpu,hy-nomas sh" if uv
           else hub.model_entry("hy-nomas")["install"])
    assert err == ["mashbox: run hy-nomas needs " + ", ".join(missing[:-1]) + f" and {missing[-1]} (mashbox[tpu,hy-nomas]), "
                   f"which this install does not have. Install them, then run the same command again: {fix}"], err
    monkeypatch.setattr(app, "run_stub", lambda a: 0)
    assert cli.main(["run", "hy-nomas", "--stub"]) == 0
    monkeypatch.setattr(weights, "check_disk", lambda log: None)
    monkeypatch.setattr(weights, "download_all", lambda log: {})
    monkeypatch.setattr(upstream, "source_dir", lambda log: "src")
    assert cli.main(["download", "hy-nomas"]) == 0


def test_run_with_the_hy_nomas_extra_installed_goes_on_to_the_model(monkeypatch):
    """With every distribution of the extra installed, the check passes and `run` reaches the model as before."""
    from mashbox import cli
    from mashbox.models.hy_nomas import app
    monkeypatch.setattr(cli, "installed", lambda d: True, raising=False)
    seen = []
    monkeypatch.setattr(app, "run", lambda a: seen.append(a.ui) or 0)
    assert cli.main(["run", "hy-nomas", "--ui"]) == 0 and seen == [True]
