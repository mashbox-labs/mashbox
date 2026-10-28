"""Upstream weights for Hy-Nomas: downloaded from Hugging Face at pinned revisions, checked by sha256, merged at load.

Nothing here is re-uploaded by us. The files and their sha256 (the Hugging Face LFS ids at the pinned revisions) are in
upstream.json next to this file:
  * Tencent's distilled DiT (fp32) and the cinematic LoRA adapter from tencent/Hy-Nomas; the adapter is merged into the
    DiT at load the way Tencent's own scripts/train/merge_lora.py does it: W' = W + (alpha / rank) * (B @ A) in fp32,
    then every weight is rounded to bf16 on the chips;
  * the pixel decoder (VAE/PixelUMM_FT, decoder tensors only are read);
  * the text encoder Qwen2.5-VL-7B-Instruct, only the shards holding the 26 layers Tencent uses.
"""
import json
import os
import shutil
from importlib import resources

import numpy as np

from . import dit as M


def pins():
    return json.loads(resources.files(__package__).joinpath('upstream.json').read_text())


def _verify(path, sha256, log):
    from ... import hub
    if os.environ.get('MASHBOX_VERIFY', '1') == '0':
        return path
    return hub.verify(path, sha256, log)


def fetch(repo_key, name, log=print):
    """Local path of one pinned upstream file (downloaded the first time, sha256 checked)."""
    from ... import hub
    p = pins()
    repo = p['repos'][repo_key]
    row = next(f for f in repo['files'] if f['name'] == name)
    path = hub.fetch(repo['hf'], name, repo['revision'])
    return _verify(path, row['sha256'], log)


def needed_bytes():
    p = pins()
    return sum(f['bytes'] for r in p['repos'].values() for f in r['files'])


def check_disk(log=print):
    """Stop early, with numbers, when the Hugging Face cache disk cannot hold the downloads still missing."""
    from huggingface_hub import constants, try_to_load_from_cache
    d = constants.HF_HUB_CACHE
    os.makedirs(d, exist_ok=True)
    free = shutil.disk_usage(d).free
    need = 0                                                # bytes of the files not downloaded yet
    for repo in pins()['repos'].values():
        for f in repo['files']:
            if not isinstance(try_to_load_from_cache(repo['hf'], f['name'], revision=repo['revision']), str):
                need += f['bytes']
    if free < need:
        log(json.dumps(dict(event='disk', free_gb=round(free / 1e9, 1), missing_gb=round(need / 1e9, 1), all_files_gb=round(needed_bytes() / 1e9, 1),
                            cache=d, note='set HF_HUB_CACHE to a bigger disk (on a TPU VM: /dev/shm/hf) before the download')))
    return free, need


def download_all(log=print):
    p = pins()
    out = {}
    for key, repo in p['repos'].items():
        for f in repo['files']:
            out[key + ':' + f['name']] = fetch(key, f['name'], log)
    return out


def text_encoder_dir(log=print):
    """Folder with the text encoder's config, tokenizer and the shards it needs (the Hugging Face snapshot folder at the
    pinned revision; other files of that repo are never downloaded)."""
    p = pins()['repos']['text_encoder']
    paths = [fetch('text_encoder', f['name'], log) for f in p['files']]
    return os.path.dirname(paths[0])


# ------------------------------------------------------------------ LoRA merge at load (Tencent's math)
def _stem(key, suffix):
    stem = key[: -len(suffix)]
    return stem.replace('._checkpoint_wrapped_module.', '.').replace('_checkpoint_wrapped_module.', '')


class MergedState:
    """Mapping over the DiT state dict that merges the LoRA adapter into each target weight when it is read.

    Reading one tensor at a time keeps host memory at the stacked parameters plus one tensor."""

    def __init__(self, dit_path, adapter_path=None, adapter_config=None):
        self.base = M.st_read(dit_path)
        self.pairs, self.scale, self.adapter = {}, 0.0, {}
        if adapter_path:
            cfg = json.load(open(adapter_config))
            rank, alpha = int(cfg['rank']), float(cfg['alpha'])
            if rank <= 0 or alpha <= 0:
                raise ValueError('invalid LoRA rank/alpha')
            self.scale, self.rank = alpha / rank, rank
            self.adapter = M.st_read(adapter_path)
            for k in self.adapter:
                if k.endswith('.lora_A.weight'):
                    self.pairs.setdefault(_stem(k, '.lora_A.weight') + '.weight', {})['A'] = k
                elif k.endswith('.lora_B.weight'):
                    self.pairs.setdefault(_stem(k, '.lora_B.weight') + '.weight', {})['B'] = k
                else:
                    raise ValueError('unexpected adapter tensor: ' + k)
            bad = [k for k, v in self.pairs.items() if set(v) != {'A', 'B'} or k not in self.base]
            if bad:
                raise ValueError('adapter targets missing or incomplete: %s' % bad[:5])
        self.merged = 0

    def __getitem__(self, k):
        w = self.base[k]
        pr = self.pairs.get(k)
        if pr is None:
            return w
        a, b = self.adapter[pr['A']], self.adapter[pr['B']]
        if w.ndim != 2 or a.shape != (self.rank, w.shape[1]) or b.shape != (w.shape[0], self.rank):
            raise ValueError('LoRA/base shape mismatch for ' + k)
        self.merged += 1
        delta = np.matmul(b.astype(np.float32), a.astype(np.float32)) * np.float32(self.scale)
        return w.astype(np.float32) + delta

    def __contains__(self, k):
        return k in self.base

    def keys(self):
        return self.base.keys()


def dit_params(dit_path, adapter_path=None, adapter_config=None, arch=M.ARCH):
    """(top, blocks, acts) f32 numpy with the adapter merged, plus a count of merged weights."""
    sd = MergedState(dit_path, adapter_path, adapter_config)
    params = M.build_params(sd, arch)
    return params, dict(merged=sd.merged, targets=len(sd.pairs), scale=sd.scale)
