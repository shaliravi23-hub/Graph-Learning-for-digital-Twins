"""Stage 7C support: predicting h windows ahead (h = 2 or 4) instead of 1.

History windows t-4 .. t-1, target window t+h-1. For h = 1 every function here gives exactly what the stage-5 functions give.
  sequences_h        history/target items built day by day from static windows (no history or target crosses a day)
  validation_items_h the same items with node-feature histories (as evaluate_temporal.validation_items)
  fit_transition_h   Transition probabilities P(edge at t+h-1 | state at t-1), fitted on training days only
                     (the same pseudo-count as evaluate_pilot.fit_transition)
  validate_h         evaluate_pilot.validate with the two horizon-dependent test checks adapted
"""
import collections
import numpy as np
import evaluate_pilot as ep
import evaluate_temporal as et


def sequences_h(static, L, h):
    if h == 1: return ep.make_sequences(static, L)
    groups = collections.defaultdict(list)
    for i in static['items']: groups[i['date']].append(i)
    out = []
    for date, items in sorted(groups.items()):
        items = sorted(items, key=lambda i: i['window_id'])
        if len({i['window_id'] for i in items}) != len(items): raise ValueError('Duplicate static windows')
        for j in range(L, len(items) - h + 1):
            history = items[j - L:j]; target = items[j + h - 1]
            ids = [i['window_id'] for i in history]
            if any(b - a != 1 for a, b in zip(ids, ids[1:])) or target['window_id'] - ids[-1] != h:
                raise ValueError('Static history or target has gaps')
            out.append(dict(date=date, target_window_id=target['window_id'], input_window_ids=ids,
                            A_seq=np.stack([i['A'] for i in history]), A_target=target['A'], X_target=target['X']))
    return out


def validation_items_h(static, L, h):
    if h == 1: return et.validation_items(static, L)
    lookup = {(i['date'], i['window_id']): i for i in static['items']}
    items = sequences_h(static, L, h)
    for i in items: i['X_seq'] = np.stack([lookup[(i['date'], w)]['X'] for w in i['input_window_ids']])
    return items


def fit_transition_h(train, h):
    if h == 1: return ep.fit_transition(train)
    n = len(train['items'][0]['A']); mask = ep.nonself(n); groups = collections.defaultdict(dict)
    for i in train['items']: groups[i['date']][i['window_id']] = i['A']
    den = np.zeros((2, n, n)); num = np.zeros_like(den)
    for windows in groups.values():
        ids = sorted(windows)
        if ids != list(range(ids[0], ids[0] + len(ids))): raise ValueError('Training day has gaps')
        for w in ids:
            if w + h in windows:
                a, b = windows[w], windows[w + h]
                for s in (0, 1):
                    m = (a == s) & mask; den[s] += m; num[s] += m * b
    pool = (num.sum((1, 2)) + 1) / (den.sum((1, 2)) + 2)
    return (num + pool[:, None, None]) / (den + 1)


def validate_h(train, val, test, meta, h):
    """evaluate_pilot.validate, with its two one-step test checks adapted to a target h windows ahead."""
    if h == 1: return ep.validate(train, val, test, meta)
    if meta.get('horizon') != h: raise ValueError(f'Dataset metadata says horizon {meta.get("horizon")}, expected {h}')
    by_date = collections.defaultdict(list)
    for i in test['items']: by_date[i['date']].append(i)
    view = []
    for items in by_date.values():
        items = sorted(items, key=lambda x: x['target_window_id'])
        for a, b in zip(items, items[1:]):
            if b['target_window_id'] != a['target_window_id'] + 1 or not np.array_equal(b['A_seq'][:-1], a['A_seq'][1:]):
                raise ValueError('Test targets are not consecutive or histories do not overlap')
        for k, i in enumerate(items):
            if i['target_window_id'] - i['input_window_ids'][-1] != h: raise ValueError('Test target is not h windows ahead')
            # one-step view for the stage-5 checks: the "target" is the window after the history, which is the next item's
            # last history window (the last item keeps its own target, used only for the shape and binary checks)
            nxt = items[k + 1]['A_seq'][-1] if k + 1 < len(items) else i['A_target']
            view.append(dict(i, target_window_id=i['input_window_ids'][-1] + 1, A_target=nxt))
    out = ep.validate(train, val, dict(test, items=view), meta)
    for i in test['items']:  # the real horizon targets must also be valid adjacency matrices
        a = i['A_target']
        if not np.isin(a, [0, 1]).all() or np.trace(a) != 0: raise ValueError('Invalid test target')
    out['horizon'] = h
    return out
