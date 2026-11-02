"""
Shared parsing helpers for make_vocab_table.py and wer_by_language.py.

Line format (both ref and hyp files):
    [0:00:00.0 - 0:00:01.0] (es) [FEM]: some words here
    [0:00:00.0 - 0:00:01.0] (es) [FEM] [OVERLAP]: some words here      (VTC2, occasionally)
    [0:00:00.0 - 0:00:01.0] (es) [FEM]: some words here[es]            (VTC2, trailing langdetect tag)

--ref-tag paren    -> language comes from the (xx) group
--hyp-tag trailing -> language comes from the [xx] tag glued to the end of the text
"""

import os
import re
import unicodedata

HEADER_RE = re.compile(
    r'^\[[^\]]*\]\s*\(([^)]*)\)\s*\[([^\]]+)\]\s*(?:\[OVERLAP\]\s*)?:\s*(.*)$'
)
TRAIL_TAG_RE = re.compile(r'^(.*?)\s*\[(\w+)\]\s*$')

ANNOTATION_RE = re.compile(r'\[.*?\]')
PLUS_MARKERS_RE = re.compile(r'\+//|\+<|\+\.\.\.')
AT_TAG_RE = re.compile(r'@\w+')
AMP_TAG_RE = re.compile(r'&\S+')
XXX_RE = re.compile(r'\bxxx\b', re.IGNORECASE)
STRIP_TOKEN_PUNCT_RE = re.compile(r'^\W+|\W+$', re.UNICODE)


def discover_clips(root, filename_suffix):
    """Yield (recording, clip, path) for every combined tagged file under root.

    root/<recording>/<clip>/<clip><filename_suffix>
    """
    for recording in sorted(os.listdir(root)):
        rec_path = os.path.join(root, recording)
        if not os.path.isdir(rec_path):
            continue
        for clip in sorted(os.listdir(rec_path)):
            clip_path = os.path.join(rec_path, clip)
            if not os.path.isdir(clip_path):
                continue
            fname = f"{clip}{filename_suffix}"
            fpath = os.path.join(clip_path, fname)
            if os.path.isfile(fpath):
                yield recording, clip, fpath


def parse_line(line, tag_mode):
    """Parse one transcript line.

    Returns (speaker, language, text) or None if the line doesn't match the
    header format, or (trailing mode) carries no per-line language tag.
    """
    line = line.rstrip('\n')
    if not line.strip():
        return None
    m = HEADER_RE.match(line)
    if not m:
        return None
    paren_lang, speaker, rest = m.group(1), m.group(2), m.group(3)
    rest = rest.strip()

    if tag_mode == 'paren':
        language = paren_lang.strip().lower()
        text = rest
    elif tag_mode == 'trailing':
        tm = TRAIL_TAG_RE.match(rest)
        if not tm:
            return None
        text = tm.group(1).strip()
        language = tm.group(2).strip().lower()
    else:
        raise ValueError(f"unknown tag_mode: {tag_mode}")

    if not language:
        return None
    return speaker, language, text


def clean_tokens(text):
    """Strip CHAT-style annotation markup; return list of raw (still-cased,
    still-punctuated) word tokens."""
    t = ANNOTATION_RE.sub('', text)
    t = PLUS_MARKERS_RE.sub('', t)
    t = AT_TAG_RE.sub('', t)
    t = AMP_TAG_RE.sub('', t)
    t = XXX_RE.sub('', t)
    t = t.replace('{', ' ').replace('}', ' ')
    t = t.replace('<', ' ').replace('>', ' ')
    t = t.replace('+', ' ').replace('-', ' ')
    return [w for w in t.split() if w]


def normalize_type(token):
    """Lowercase + strip surrounding punctuation, for vocabulary/type counting."""
    w = STRIP_TOKEN_PUNCT_RE.sub('', token)
    return w.lower()


def remove_diacritics(text):
    nfd = unicodedata.normalize('NFD', text)
    return ''.join(c for c in nfd if unicodedata.category(c) != 'Mn')


def wer_clean_tokens(tokens):
    """Further clean already-annotation-stripped tokens for WER scoring:
    strip diacritics + punctuation, lowercase. Mirrors evaluation_step4.py's
    clean_text() but applied per-token so it can be bucketed by language."""
    out = []
    for tok in tokens:
        t = remove_diacritics(tok)
        t = re.sub(r'[^A-Za-z0-9]', '', t)
        t = t.lower()
        if t:
            out.append(t)
    return out


CANON_RE = re.compile(r'^(.*)_(\d+)$')


def canonical_key(clip_folder):
    """Derive (recording, clip_label) from the clip folder name alone, e.g.
    'P06_pre_030' -> ('P06_pre', '030').

    The two roots (ref/hyp) don't always name the *parent* recording folder
    the same way (e.g. ref has 'P04_pre_1', hyp has
    'P04_pre' for the same recording) even though the clip folder
    name itself is identical on both sides. Deriving the key from the clip
    folder's trailing numeric suffix sidesteps that mismatch entirely.
    """
    m = CANON_RE.match(clip_folder)
    if not m:
        return clip_folder, clip_folder
    return m.group(1), m.group(2)


def anonymize_recordings(recordings):
    """Stable REC001..REC0NN mapping, sorted for determinism."""
    ordered = sorted(set(recordings))
    return {rec: f"{i + 1:03d}" for i, rec in enumerate(ordered)}
