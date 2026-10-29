# Copyright 2026 MASHBOX Labs
# SPDX-License-Identifier: Apache-2.0
"""Frame-0 subject mask and the query points the tracker follows.

The mask comes from your own PNG (white = subject) or from rembg (MIT, optional install)
with the BiRefNet general model (MIT). We keep the largest connected region, erode it by
2 pixels to stay off the silhouette edge (depth bleeds there), and sample N pixels from it
with a fixed seed.
"""
import numpy as np


def subject_mask(frame0, mask_path=None, rembg_model="birefnet-general"):
    """frame0: uint8 [H, W, 3]. Returns bool [H, W]."""
    from scipy import ndimage
    H, W = frame0.shape[:2]
    if mask_path:
        from PIL import Image
        img = Image.open(mask_path).convert("L")
        if img.size != (W, H):
            img = img.resize((W, H), Image.NEAREST)
        m = np.asarray(img) > 127
    else:
        try:
            from rembg import new_session, remove
        except ImportError as e:
            raise SystemExit("rembg is not installed. Install it (pip install \"rembg[gpu]\") "
                             "or pass --mask frame0_mask.png") from e
        from PIL import Image
        alpha = np.array(remove(Image.fromarray(frame0), session=new_session(rembg_model), only_mask=True)) / 255.0
        m = alpha > 0.5
    lab, nl = ndimage.label(m)
    if nl > 1:
        sizes = ndimage.sum(m, lab, range(1, nl + 1))
        m = lab == (1 + int(np.argmax(sizes)))
    m = ndimage.binary_erosion(m, iterations=2)
    if m.sum() < 8:
        raise SystemExit("subject mask has fewer than 8 pixels after cleanup; check frame 0 or pass --mask")
    return m


def sample_queries(mask, n, seed=0):
    """Returns [n', 3] float32 rows [frame=0, x_px, y_px] at pixel centres, n' = min(n, mask pixels)."""
    ys, xs = np.nonzero(mask)
    rng = np.random.default_rng(seed)
    sel = rng.choice(len(ys), min(n, len(ys)), replace=False)
    return np.stack([np.zeros(len(sel)), xs[sel] + 0.5, ys[sel] + 0.5], 1).astype(np.float32)
