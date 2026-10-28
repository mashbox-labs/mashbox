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
