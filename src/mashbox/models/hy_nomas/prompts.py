"""Prompts in Tencent's structured Hy-Nomas format.

A world is two texts: seg1 (style, view, subject, scene, first 5 s) for the first window and seg2 ("compared to the previous
segment ...") for every later window. A timed action is seg2 with its contained_actions split in three: the base action
before, the new line from ``start`` for ``dur`` seconds, the base action after. Tencent's tokenizer turns those timings
into the per-action text mask.
"""
import json
import re

STYLES = ['real-world footage', 'animation/anime', 'CG', 'video game']
VIEWS = ['first-person', 'second-person', 'third-person follow', 'third-person observer']
PRESET_ACTIONS = ['leaps high into the air', 'dives steeply, head first', 'darts forward in a sudden burst of speed',
                  'climbs up quickly', 'runs forward fast', 'crouches low to the ground']


def ts(x):
    m = int(x // 60)
    return '%02d:%06.3f' % (m, x - 60 * m)


def act_line(i, text):
    return '<Action%d><|Subject1|> %s</Action%d>.' % (i, text, i)


def clean_text(s, n=300):
    s = re.sub(r'[\x00-\x1f<>{}|]', ' ', str(s or ''))
    s = re.sub(r'\s+', ' ', s).strip()
    return s[:n].rstrip(' .')


def world_from_form(f):
    """Form fields (name, look, scene, camera, action, style, view) -> (seg1, seg2) JSON strings."""
    name, look, scene = clean_text(f.get('name'), 80), clean_text(f.get('look'), 300), clean_text(f.get('scene'), 400)
    cam = clean_text(f.get('camera') or 'The camera follows from behind at a steady distance', 160)
    act = clean_text(f.get('action'), 160) or 'moves slowly forward'
    style = f.get('style') if f.get('style') in STYLES else 'real-world footage'
    view = f.get('view') if f.get('view') in VIEWS else 'third-person follow'
    if not name or not scene:
        raise ValueError('subject and scene are needed')
    acts = [{"start_time": "00:00.000", "end_time": "00:05.000", "action_description": act_line(1, act)}]
    seg1 = {"Style": style, "perspective_view": view,
            "subject_list": [{"subject_id": "<|Subject1|>", "display_name": name, "subject_attribute": (look + '.') if look else name + '.'}],
            "segments": [{"visual_description_long": scene + '. <Camera1>' + cam + '</Camera1>.', "contained_actions": acts}]}
    seg2 = {"segments": [{"description_change": "Compared to the previous segment, the surroundings stay the same: "
                          + scene[0].lower() + scene[1:] + '. <Camera1>' + cam + '</Camera1>.', "contained_actions": acts}]}
    return json.dumps(seg1), json.dumps(seg2)


def base_action(seg2):
    d = json.loads(seg2)['segments'][0]['contained_actions'][0]['action_description']
    return re.sub(r'^<Action\d+><\|Subject1\|>\s*|</Action\d+>\.?$', '', d).strip()


def display_name(seg1):
    try:
        return json.loads(seg1)['subject_list'][0]['display_name']
    except (ValueError, KeyError, IndexError, TypeError):
        return ''


def action_text(seg1, text, subject=True):
    """Typed action -> the line that goes in the prompt ('the eagle dives' -> 'dives': the subject token stands for it)."""
    t = clean_text(text, 160)
    if not t:
        raise ValueError('empty action')
    if subject:
        words = set(w.lower() for w in re.findall(r'[A-Za-z]+', display_name(seg1)) if len(w) > 2)
        m = re.match(r'^(?:the|a|an|my|our)?\s*([A-Za-z]+)\s+(.*)$', t, re.I)
        if m and m.group(1).lower() in words:
            t = m.group(2)
    return t


def action_seg2(seg2_base, line, start, dur, subject=True):
    """seg2 with contained_actions = [base 0..start] + [line start..start+dur] + [base ..5]. -> (seg2, (start, end))."""
    d = json.loads(seg2_base)
    seg = d['segments'][0]
    b = base_action(seg2_base)
    start = max(0.0, min(4.5, round(float(start) * 2) / 2))
    end = min(5.0, start + max(0.5, round(float(dur) * 2) / 2))
    acts, i = [], 1
    for s0, e0, txt, subj in ((0.0, start, b, True), (start, end, line, subject), (end, 5.0, b, True)):
        if e0 - s0 >= 0.25:
            desc = act_line(i, txt) if subj else '<Action%d>%s</Action%d>.' % (i, txt, i)
            acts.append({"start_time": ts(s0), "end_time": ts(e0), "action_description": desc})
            i += 1
    seg['contained_actions'] = acts
    return json.dumps(d), (start, end)


def check_json_world(seg1, seg2):
    j1, j2 = json.loads(seg1), json.loads(seg2)
    if not ('subject_list' in j1 and 'segments' in j1 and 'segments' in j2):
        raise ValueError('seg1 needs subject_list and segments, seg2 needs segments')
