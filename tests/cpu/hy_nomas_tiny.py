"""Tiny Hy-Nomas checkpoints for CPU tests: Tencent's tensor names and layout, random values, a few MB.

Same architecture as the real model (double-stream blocks, camera action module, text refiner, final layer) with small
sizes. Used by the CPU tests to check shapes, padding, the LoRA merge and the window loop without the 34 GB weights.
"""
import json
import struct

import numpy as np


def write_safetensors(path, tensors):
    """{name: f32 numpy} -> safetensors file (F32), no extra dependency."""
    header, off, blobs = {}, 0, []
    for k in sorted(tensors):
        a = np.ascontiguousarray(tensors[k], np.float32)
        header[k] = dict(dtype='F32', shape=list(a.shape), data_offsets=[off, off + a.nbytes])
        off += a.nbytes
        blobs.append(a.tobytes())
    h = json.dumps(header).encode()
    h += b' ' * ((8 - len(h) % 8) % 8)
    with open(path, 'wb') as f:
        f.write(struct.pack('<Q', len(h)))
        f.write(h)
        for b in blobs:
            f.write(b)


def dit_state(arch, seed=0):
    """Random DiT state dict with Tencent's names for an Arch (hidden, heads, ff, layers, text_dim, cam_heads)."""
    r = np.random.RandomState(seed)
    D, FF, TXD, CKH = arch.hidden, arch.ff, arch.text_dim, arch.cam_heads * arch.cam_head_dim
    sd = {}

    def lin(name, out, inp, scale=1.0, bias=True):
        sd[name + '.weight'] = (r.randn(out, inp) * scale / np.sqrt(inp)).astype(np.float32)
        if bias:
            sd[name + '.bias'] = (r.randn(out) * 0.02).astype(np.float32)

    def norm(name, n):
        sd[name + '.weight'] = (1 + 0.1 * r.randn(n)).astype(np.float32)
    for i in range(arch.layers):
        for s in ('img', 'txt'):
            p = 'double_blocks.%d.%s_' % (i, s)
            lin(p + 'mod.linear', 6 * D, D, 0.2)
            for n in ('attn_q', 'attn_k', 'attn_v', 'attn_proj'):
                lin(p + n, D, D)
            lin(p + 'mlp.fc1', FF, D)
            lin(p + 'mlp.fc2', D, FF)
            norm(p + 'attn_q_norm', arch.head_dim)
            norm(p + 'attn_k_norm', arch.head_dim)
        a = 'action_modules.%d.' % i
        lin(a + 'img_attn_q', CKH, D, bias=False)
        norm(a + 'key_attn_q_norm', arch.cam_head_dim)
        norm(a + 'key_attn_k_norm', arch.cam_head_dim)
        lin(a + 'keyboard_attn_kv', 2 * CKH, 128, bias=False)
        lin(a + 'keyboard_embed.0', 128, 64)
        lin(a + 'keyboard_embed.2', 128, 128)
        lin(a + 'proj_keyboard', D, CKH, 0.5, bias=False)
    for n in ('c2ws_hidden_states_layer1', 'c2ws_hidden_states_layer2', 'extrinsics_embedding_camctrl'):
        lin(n, 64, 64)
    lin('final_layer.adaLN_modulation.1', 2 * D, D, 0.2)
    lin('final_layer.linear', arch.latent_ch, D)
    sd['img_in.proj.weight'] = (r.randn(D, arch.latent_ch, 1, 1, 1) / np.sqrt(arch.latent_ch)).astype(np.float32)
    sd['img_in.proj.bias'] = (r.randn(D) * 0.02).astype(np.float32)
    for m in ('time_in.mlp', 'txt_in.t_embedder.mlp'):
        lin(m + '.0', D, 256)
        lin(m + '.2', D, D)
    lin('txt_in.c_embedder.linear_1', D, TXD)
    lin('txt_in.c_embedder.linear_2', D, D)
    lin('txt_in.input_embedder', D, TXD)
    for j in range(2):
        b = 'txt_in.individual_token_refiner.blocks.%d.' % j
        lin(b + 'adaLN_modulation.1', 2 * D, D, 0.2)
        lin(b + 'mlp.fc1', 4 * D, D)
        lin(b + 'mlp.fc2', D, 4 * D)
        lin(b + 'self_attn_proj', D, D)
        lin(b + 'self_attn_qkv', 3 * D, D)
        for n in ('norm1', 'norm2'):
            norm(b + n, D)
            sd[b + n + '.bias'] = (0.02 * r.randn(D)).astype(np.float32)
    return sd


LORA_TARGETS = ['double_blocks.*.%s_%s' % (s, n) for s in ('img', 'txt')
                for n in ('attn_q', 'attn_k', 'attn_v', 'attn_proj', 'mlp.fc1', 'mlp.fc2', 'mod.linear')] + \
               ['action_modules.*.img_attn_q', 'action_modules.*.keyboard_attn_kv', 'action_modules.*.proj_keyboard']


def lora_state(sd, layers, rank=4, seed=1):
    """Random LoRA adapter (Tencent's adapter.safetensors names) for every target of the real adapter_config.json."""
    r = np.random.RandomState(seed)
    out = {}
    for i in range(layers):
        for t in LORA_TARGETS:
            stem = t.replace('*', str(i))
            w = sd[stem + '.weight']
            out[stem + '.lora_A.weight'] = (r.randn(rank, w.shape[1]) * 0.05).astype(np.float32)
            out[stem + '.lora_B.weight'] = (r.randn(w.shape[0], rank) * 0.05).astype(np.float32)
    return out


def lora_config(rank=4, alpha=8.0):
    return dict(format_version=1, rank=rank, alpha=alpha, dropout=0.0, targets=LORA_TARGETS)
