"""Tencent's own Hy-Nomas Python code, used unchanged for three things: the text prompt template and tokenizer crop, the
timed-action offsets inside a prompt, and the camera input normalization.

The code is downloaded once from GitHub at the commit pinned in upstream.json, only its hymm/ and configs/ folders are
kept, and their combined hash is checked before anything is imported. It runs on the host CPU (torch CPU, transformers).
"""
import ast
import hashlib
import importlib.util
import io
import json
import logging
import os
import sys
import tarfile
import textwrap
import types
import urllib.request
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from .weights import pins

KEEP = ('hymm', 'configs')
TEXT_LEN = 1280


def tree_hash(root):
    rows = []
    for top in KEEP:
        for dp, dn, fn in os.walk(os.path.join(root, top)):
            # Python writes __pycache__ folders next to the code the first time it is imported: they are not Tencent's code
            dn[:] = sorted(x for x in dn if x != '__pycache__')
            for f in sorted(fn):
                p = os.path.join(dp, f)
                with open(p, 'rb') as fh:
                    rows.append((os.path.relpath(p, root), hashlib.sha256(fh.read()).hexdigest()))
    rows.sort()
    return hashlib.sha256(''.join('%s %s\n' % (h, p) for p, h in rows).encode()).hexdigest(), len(rows)


def source_dir(log=print):
    """Path of Tencent's code at the pinned commit (downloaded and checked the first time)."""
    from ... import hub
    c = pins()['code']
    d = os.path.join(hub.cache_home(), 'upstream', 'tencent-hy-nomas-code', c['commit'])
    if os.path.isdir(d):
        got, _ = tree_hash(d)
        if got == c['tree_sha256']:
            return d
        raise SystemExit('%s: code hash %s does not match the pinned %s; delete the folder and run again' % (d, got, c['tree_sha256']))
    url = c['archive'].format(commit=c['commit'])
    log(json.dumps(dict(event='download', what='Tencent Hy-Nomas code', commit=c['commit'][:12])))
    blob = urllib.request.urlopen(url, timeout=120).read()
    tmp = d + '.part'
    os.makedirs(tmp, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(blob), mode='r:gz') as tf:
        for m in tf.getmembers():
            parts = m.name.split('/', 1)
            if len(parts) < 2 or not parts[1].split('/', 1)[0] in KEEP:
                continue
            rel = os.path.normpath(parts[1])
            if rel.startswith('..') or os.path.isabs(rel) or not (m.isfile() or m.isdir()):
                continue
            out = os.path.join(tmp, rel)
            if m.isdir():
                os.makedirs(out, exist_ok=True)
                continue
            os.makedirs(os.path.dirname(out), exist_ok=True)
            with open(out, 'wb') as fh:
                fh.write(tf.extractfile(m).read())
    got, n = tree_hash(tmp)
    if got != c['tree_sha256']:
        raise SystemExit('Tencent code at %s: hash %s (%d files) does not match the pinned %s' % (c['commit'][:12], got, n, c['tree_sha256']))
    os.replace(tmp, d)
    return d


def _on_path(src):
    if src not in sys.path:
        sys.path.insert(0, src)
    # Tencent's code logs with loguru; stand in with logging if absent. Check sys.modules first: once the stand-in is there,
    # find_spec('loguru') raises ValueError (a module without __spec__), so a second call must not ask again.
    if 'loguru' not in sys.modules and importlib.util.find_spec('loguru') is None:
        sys.modules['loguru'] = types.SimpleNamespace(logger=logging.getLogger('hymm'))


# ------------------------------------------------------------------ text: template, crop, action offsets
class TencentText:
    """Tencent's TextEncoder (configs/experiment/inference.yaml) built with a stand-in model, so its tokenizer, prompt
    template and crop run exactly as upstream without loading the torch weights; plus Tencent's action offset helpers."""

    def __init__(self, src, encoder_dir, frames=30):
        import torch
        import yaml
        _on_path(src)
        self.frames = frames
        cfg = yaml.safe_load(open(os.path.join(src, 'configs/experiment/inference.yaml')).read())
        te = cfg['model']['text_encoder']
        if te['hidden_state_skip_layer'] != 2 or te['text_len'] != TEXT_LEN or te['text_encoder_precision'] != 'bf16':
            raise RuntimeError('unexpected text encoder config: %s' % te)
        self.fps = float(cfg.get('inference', {}).get('video_fps', 24.0))
        spec = importlib.util.spec_from_file_location('hy_nomas_text_encoder', os.path.join(src, 'hymm/models/text_encoder/encoder.py'))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        from configs.constants import PROMPT_TEMPLATE
        stub = SimpleNamespace(dtype=torch.bfloat16, device=torch.device('cpu'))
        lg = logging.getLogger('hy-nomas-text')
        lg.setLevel(logging.WARNING)
        with patch.object(mod, 'load_text_encoder', return_value=(stub, str(encoder_dir))):
            self.enc = mod.TextEncoder(
                text_encoder_type=te['name'], max_length=te['text_len'], text_encoder_precision=te['text_encoder_precision'],
                text_encoder_path=str(encoder_dir), tokenizer_type=te['tokenizer'], tokenizer_path=str(encoder_dir),
                use_attention_mask=te['use_attention_mask'], prompt_template=deepcopy(PROMPT_TEMPLATE[te['prompt_template']]),
                prompt_template_video=deepcopy(PROMPT_TEMPLATE[te['prompt_template_video']]),
                hidden_state_skip_layer=te['hidden_state_skip_layer'], apply_final_norm=False, logger=lg, device=torch.device('cpu'))
        self.crop = int(self.enc.prompt_template_video.get('crop_start'))
        self._helpers(src)

    def _helpers(self, src):
        def defs(path, names):
            s = open(path).read()
            got = {}
            for node in ast.walk(ast.parse(s)):
                if isinstance(node, ast.FunctionDef) and node.name in names and node.name not in got:
                    got[node.name] = ast.get_source_segment(s, node)
            if set(got) != set(names):
                raise RuntimeError('%s: missing %s' % (path, set(names) - set(got)))
            return got
        ns = {'Optional': __import__('typing').Optional, 'List': __import__('typing').List}
        exec(defs(os.path.join(src, 'hymm/inference.py'), ['_parse_timestamp'])['_parse_timestamp'], ns)
        sys.modules.setdefault('hymm.inference', types.SimpleNamespace(_parse_timestamp=ns['_parse_timestamp']))
        names = ['_compute_action_text_offsets', '_compute_action_video_windows', '_compute_action_start']
        for v in defs(os.path.join(src, 'hymm/pipeline/base.py'), names).values():
            exec(textwrap.dedent(v), ns)
        self.fn = types.SimpleNamespace(**{k: ns[k] for k in names})
        self.me = SimpleNamespace(text_encoder=self.enc, t_ratio=4, video_fps=self.fps,
                                  args=SimpleNamespace(text_len=TEXT_LEN, inference=SimpleNamespace(video_fps=self.fps), video_fps=self.fps))

    def tokens(self, prompt):
        """-> ids int32 [crop + 1280], attention mask bool (right padding)."""
        t = self.enc.text2tokens([prompt], data_type='video', max_length=TEXT_LEN)
        ids = t['input_ids'][0].numpy().astype(np.int32)
        am = t['attention_mask'][0].numpy().astype(bool)
        if ids.shape != (TEXT_LEN + self.crop,):
            raise RuntimeError('token shape %s' % (ids.shape,))
        return ids, am

    def offsets(self, prompt, valid_total):
        offs = self.fn._compute_action_text_offsets(self.me, prompt)
        win = self.fn._compute_action_video_windows(self.me, prompt, 0, self.frames) if offs is not None else None
        a0 = self.fn._compute_action_start(self.me, prompt)
        n = valid_total - self.crop
        return dict(valid_post_crop=n, offs=None if offs is None else [min(int(o), n) for o in offs], win=win, action_start=a0)


# ------------------------------------------------------------------ camera
def camera_module(src):
    """Tencent's camera helpers: (action_sequence_to_extrinsics, CameraConfig, prepare_camera_inputs). The pipeline
    package is stood in by empty modules so only the camera code is imported. Action labels are not used by this model's
    camera path, so their per-frame fits are skipped (labels of the same length, all 0; extrinsics unchanged)."""
    _on_path(src)
    from hymm.models.camera.geometry.keyboard_mouse_to_extrinsics import action_sequence_to_extrinsics, CameraConfig
    for m in ('hymm.pipeline', 'hymm.pipeline.base', 'hymm.pipeline.solvers', 'hymm.pipeline.solvers.euler'):
        sys.modules.setdefault(m, types.ModuleType(m))
    sys.modules['hymm.pipeline'].__path__ = [os.path.join(src, 'hymm', 'pipeline')]
    sys.modules['hymm.pipeline.base'].BasePipeline = type('BasePipeline', (), {})
    sys.modules['hymm.pipeline.base'].align_to = None
    sys.modules['hymm.pipeline.solvers.euler'].EulerSolver = type('EulerSolver', (), {})
    spec = importlib.util.spec_from_file_location('hymm.pipeline.bidirectional', os.path.join(src, 'hymm', 'pipeline', 'bidirectional.py'))
    bd = importlib.util.module_from_spec(spec)
    sys.modules['hymm.pipeline.bidirectional'] = bd
    spec.loader.exec_module(bd)
    import hymm.models.camera.geometry.discretize_poses as dp

    def labels_only(pose_dict, mode='4t4r_robust', downsample_ratio=1, normalize_displacement=False, **kw):
        if mode != '4t4r_robust' or downsample_ratio != 1 or normalize_displacement or kw:
            raise RuntimeError('unexpected label request')
        return None, None, np.zeros(len(pose_dict), np.int64)
    dp.discretize_poses = labels_only
    return action_sequence_to_extrinsics, CameraConfig, bd.prepare_camera_inputs
