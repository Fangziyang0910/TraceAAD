"""Atomically apply content-addressed edits to one host-bound source snapshot."""

import re

from traceaad.common.delivery import DeliveryError, SourceError, extract_idea, parse_response, validate_source

MARKERS = re.compile(r'(?m)^(<<<<<<< SEARCH|=======|>>>>>>> REPLACE)[ \t]*\r?$')
MARKER_LIKE = re.compile(r'(?m)^[ \t]*(?:<{3,}[ \t]*SEARCH|={5,}|>{3,}[ \t]*REPLACE)[^\n]*$')
EDITS = re.compile(r'(?im)^[ \t]*(?:\*\*)?Edits:(?:\*\*)?[ \t]*$')


def clean_response(text):
    return re.sub(r'<think>.*?</think>', '', text, flags=re.S | re.I) if isinstance(text, str) else ''


def mode(text):
    text = clean_response(text)
    return 'edit' if EDITS.search(text) or MARKERS.search(text) else 'full'


def edit_design(text):
    text = clean_response(text)
    start = EDITS.search(text) or MARKERS.search(text)
    return extract_idea(text[:start.start()] if start else text)


def apply_edits(base, payload):
    """All searches address the original base, uniquely and without overlap."""
    markers = list(MARKERS.finditer(payload))
    if [m.span() for m in MARKER_LIKE.finditer(payload)] != [m.span() for m in markers]:
        raise DeliveryError('edit: malformed marker; use exact SEARCH/REPLACE markers for every block')
    if not markers or len(markers) % 3:
        raise DeliveryError('edit: incomplete SEARCH/REPLACE block; no changes applied')
    spans = []
    for index in range(0, len(markers), 3):
        start, middle, end = markers[index:index + 3]
        if [m[1] for m in (start, middle, end)] != ['<<<<<<< SEARCH', '=======', '>>>>>>> REPLACE']:
            raise DeliveryError('edit: invalid marker order; no changes applied')
        # Remove only the delimiter line breaks, preserving indentation and source whitespace.
        search = payload[start.end():middle.start()].removeprefix('\n').removesuffix('\n')
        replacement = payload[middle.end():end.start()].removeprefix('\n').removesuffix('\n')
        if not search.strip():
            raise DeliveryError(f'edit block {index // 3 + 1}: empty SEARCH; include a unique existing anchor')
        position = base.find(search)
        if position < 0:
            raise DeliveryError(f'edit block {index // 3 + 1}: SEARCH not found in the bound base; copy its exact text')
        if base.find(search, position + 1) >= 0:
            raise DeliveryError(f'edit block {index // 3 + 1}: SEARCH is ambiguous; include more surrounding code')
        spans.append((position, position + len(search), replacement))
    spans.sort()
    if any(left[1] > right[0] for left, right in zip(spans, spans[1:])):
        raise DeliveryError('edit: overlapping SEARCH regions; combine related edits into one block')
    updated = base
    for start, end, replacement in reversed(spans):
        updated = updated[:start] + replacement + updated[end:]
    return updated, len(spans)


def parse_candidate(text, finish_reason, template, base=None):
    if mode(text) == 'full':
        code, idea, delivery = parse_response(text, finish_reason, template)
        return code, idea, {**delivery, 'mode': 'full'}
    if finish_reason != 'stop' or not isinstance(text, str):
        raise DeliveryError('edit: incomplete or non-text completion; no changes applied')
    text = clean_response(text).replace('\r\n', '\n')
    if '<think>' in text.lower():
        raise DeliveryError('edit: unclosed reasoning block; no changes applied')
    if base is None:
        raise DeliveryError('edit: no bound program; initialization requires complete Code')
    label = EDITS.search(text)
    payload = text[label.end():] if label else text
    if re.search(r'(?im)^\s*Code\s*:\s*$', text):
        raise DeliveryError('edit: submit Edits or complete Code, not both')
    code, count = apply_edits(base, payload)
    idea = edit_design(text)
    try:
        validate_source(code, template)
    except (SyntaxError, ValueError) as exc:
        raise SourceError(str(exc), code) from exc
    return code, idea, {'mode': 'edit', 'strategy': 'atomic_exact_search_replace', 'edit_count': count}
