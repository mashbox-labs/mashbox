"""Fixed text lengths for Hy-Nomas on TPU.

Every compiled program has fixed shapes. If the text length changed the shapes, each new prompt would compile new
programs (35 to 76 s on TPU v6e-4). Here every prompt is padded to one of two lengths, and the padded positions are
masked out of attention, so a new world or action reuses the programs that are already compiled.

A key is visible to a query when ``gk < 0`` or ``gk == gq``. That one rule carries Tencent's two inference text masks
(per-action text mask and head-only previous text) and the padding: padded keys get ``MASKED``, which no query has.
"""
import numpy as np

BUCKETS = (256, 512)          # text tokens after the template crop; preset worlds and actions use 108 to 251
MASKED = 1 << 30              # group id that no query row ever has


def bucket(n, sizes=BUCKETS):
    """Smallest bucket that holds n text tokens."""
    for b in sizes:
        if n <= b:
            return b
    raise ValueError("text is %d tokens; the longest supported text is %d tokens. Shorten the description." % (n, sizes[-1]))


def pad_rows(x, size):
    """Pad rows of x [n, d] with zeros up to size rows (zero rows keep every padded value finite)."""
    x = np.asarray(x)
    if x.shape[0] > size:
        raise ValueError("%d rows do not fit in %d" % (x.shape[0], size))
    return np.concatenate([x, np.zeros((size - x.shape[0],) + x.shape[1:], x.dtype)])


def text_groups(n_text, offs):
    """Group id per text token: -1 for the text head (visible to all), k for the tokens of timed action k."""
    tg = np.full(n_text, -1, np.int32)
    if offs:
        for k in range(len(offs) - 1):
            tg[offs[k]:offs[k + 1]] = k
    return tg


def rup(x, m):
    return (x + m - 1) // m * m


def layout(n_img, text_bucket, hist_img=0, prev_bucket=0, block_q=1024, block_kv=7168):
    """Rows of one attention call: (LQ, LK, nvalid). Keys are [image | text bucket | history image | previous text bucket]."""
    nvalid = n_img + text_bucket + (hist_img + prev_bucket if hist_img else 0)
    return rup(n_img + text_bucket, block_q), rup(nvalid, block_kv), nvalid


def groups(n_img, frame_tokens, frame_act, n_text, offs, text_bucket, LQ, LK, hist_img=0, n_prev=0, prev_keep=0, prev_bucket=0):
    """Group arrays (gq [LQ, 1], gk [1, LK], int32) for one attention call.

    frame_act: action index per latent frame of the window (-1 = no timed action), or None for a prompt without actions.
    offs: action offsets inside the text (head end first, text end last), or None.
    n_prev, prev_keep: real tokens of the previous window's text and how many of them stay visible (all of them, or the
    head only when that text had timed actions).
    """
    gq = np.full((LQ, 1), -1, np.int32)
    gk = np.full((1, LK), -1, np.int32)
    if frame_act is not None:
        gq[:n_img, 0] = np.repeat(np.asarray(frame_act, np.int32), frame_tokens)
    tg = text_groups(n_text, offs)
    gq[n_img:n_img + n_text, 0] = tg
    gk[0, n_img:n_img + n_text] = tg
    gk[0, n_img + n_text:n_img + text_bucket] = MASKED
    if hist_img:
        p0 = n_img + text_bucket + hist_img
        gk[0, p0 + prev_keep:p0 + prev_bucket] = MASKED
        assert prev_keep <= n_prev <= prev_bucket
    return gq, gk


def shape_count(lengths, n_img=26400, hist_img=15840):
    """Distinct attention shapes for a list of text lengths (first window and later windows)."""
    first = {layout(n_img, bucket(n)) for n in lengths}
    later = {layout(n_img, bucket(n), hist_img, bucket(p)) for n in lengths for p in lengths}
    return len(first), len(later)
