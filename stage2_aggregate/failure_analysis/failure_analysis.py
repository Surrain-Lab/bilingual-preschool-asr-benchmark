#!/usr/bin/env python3
"""
Failure analyses reported in the Results ("Sources of English insertions and empty segments",
"Transcription accuracy across pipelines") and the Methods corpus description.

Reads stage-1 outputs and the expert transcripts, and writes COUNTS ONLY (no transcript text,
no study IDs) to data/derived/failure_analysis/, which verify.py checks against the paper:

  summary.csv              metric,value  -- every scalar quoted in the paper from these analyses
  stock_phrases.csv        the 25 most frequent English utterances of VTC2 + WhisperX, with
                           pipeline / coder counts and where they fall relative to coded speech
  discarded_languages.csv  language Whisper used for the VTC2 segments the pipeline discarded
  segments_by_speaker.csv  all / empty / stock-phrase segments and English utterances by speaker

Inputs (all from a stage-1 run, e.g. <out>/output_14clips/ on the transcription machine):

  --vtc2-langdetect   vtc2/langdetect                     lines: [t0 - t1] (whisper_lang) [SPK]: text[tag]
  --nolang-langdetect vtc2_nolang_barrier/langdetect      same format, other languages kept
  --human-dir         annotated-text/  (the expert transcripts, one sub-folder per source)
  --posthoc-vtc1      whisperx_only/tagged_transcripts    post-hoc tagged files (*_<SPK>_tagged.txt)
  --posthoc-vtc2      whisperx_vtc2_posthoc               (see ../../posthoc_vtc2/README.md)
  --vtc1-seg          vtc1/rttm_txt                       classifier segments per speaker
  --vtc2-seg          vtc2/rttm_txt
  --human-wordcount   human_eval/word_count               combined_statistics.csv + *_words.txt
  --human-clean       human_eval/five_mins_benchmark      the cleaned reference (*_clean.txt)

    python failure_analysis.py --vtc2-langdetect ... --out ../../data/derived/failure_analysis

Definitions match the paper:
  * stock phrases   = the 25 most frequent English utterances (normalised: lower case, no
                      punctuation or accents); 'non-speech' = the coders recorded nothing at that
                      moment ('nothing') or only non-lexical material ('non-lexical': laughs, noises,
                      xxx); overlap language = language tag of the coder utterance overlapping most
  * language-label agreement = words whose langdetect tag equals Whisper's language
  * empty segment   = a VTC2 + WhisperX segment with no words kept
  * coder text uses the evaluation's own clean_text() (VTC2_Pipeline/evaluation.py)
"""
import argparse, csv, glob, os, re, statistics as st, string, sys, unicodedata
from collections import Counter, defaultdict

# ------------------------------------------------------------------ text helpers
def clean_text(text):
    """Verbatim from stage1_pipeline/VTC2_Pipeline/evaluation.py (the paper's scoring cleanup)."""
    text = re.sub(r'\[.*?\]', '', text)
    text = text.replace('+//', '').replace('+<', '')
    text = re.sub(r'\bxxx\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'[@&][^\s]+', '', text)
    text = text.replace('+', '').replace('-', '')
    text = re.sub(r'\bde el\b', 'del', text, flags=re.IGNORECASE)
    text = re.sub(r'\ba el\b', 'al', text, flags=re.IGNORECASE)
    text = re.sub(r'[¿¡]', '', text)
    text = ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return ' '.join(text.split()).lower()

def norm(t):
    """Normalisation for comparing short utterances (keeps apostrophes: i'm sorry)."""
    t = ''.join(c for c in unicodedata.normalize('NFD', t.lower()) if unicodedata.category(c) != 'Mn')
    return ' '.join(re.sub(r"[^\w\s']", ' ', t).split())

def clean_human(t):
    t = re.sub(r'&=\S+', ' ', t); t = re.sub(r'&[+-]\S+', ' ', t)
    t = re.sub(r'\bxxx\b|\byyy\b|\bwww\b', ' ', t, flags=re.I)
    t = re.sub(r'\[[^\]]*\]|<|>', ' ', t).replace('{', ' ').replace('}', ' ')
    return norm(re.sub(r'\b0\b', ' ', t))

ck = lambda c: re.sub(r'_1_(\d{3})$', r'_\1', re.sub(r'_1_(\d{3})_', r'_\1_', c))
sec = lambda h, m, s: int(h) * 3600 + int(m) * 60 + float(s)
TS = re.compile(r'^\[(\d+):(\d+):([\d.]+)\s*-\s*(\d+):(\d+):([\d.]+)\]\s*\(([\w-]+)\)\s*\[([^\]]+)\](\s*\[OVERLAP\])?:\s*(.*)$')
SEGLINE = re.compile(r'^\[(\d+):(\d+):([\d.]+)\s*-\s*(\d+):(\d+):([\d.]+)\]')
SPK4 = r'(KCHI|OCH|FEM|MAL)'

def overall_files(root, suffix):
    for f in glob.glob(os.path.join(root, '*', '*', '*' + suffix)):
        if not re.search('_' + SPK4 + re.escape(suffix) + '$', f):
            yield f

def load_pipeline(root):
    """clip -> list of segments (VTC2 + WhisperX or NoLang langdetect output)."""
    out = {}
    for f in overall_files(root, '_tagged_langdetect.txt'):
        clip = ck(os.path.basename(os.path.dirname(f))); segs = []
        for line in open(f, encoding='utf-8', errors='replace'):
            m = TS.match(line.strip())
            if not m: continue
            txt = m.group(10).strip(); lm = re.match(r'^(.*?)\s*\[(\w+)\]$', txt)
            text, tag = (lm.group(1).strip(), lm.group(2)) if lm else (txt, None)
            segs.append(dict(st=sec(*m.group(1, 2, 3)), en=sec(*m.group(4, 5, 6)), wl=m.group(7),
                             spk=m.group(8), ov=bool(m.group(9)), text=text, tag=tag, n=norm(text),
                             key=(m.group(8), round(sec(*m.group(1, 2, 3)), 2), round(sec(*m.group(4, 5, 6)), 2))))
        out[clip] = segs
    return out

def load_human(root):
    out = defaultdict(list)
    for f in glob.glob(os.path.join(root, '*', '*', '*.txt')):
        clip = ck(re.match(r'^(.*?_\d{3})', os.path.basename(f)).group(1))
        for line in open(f, encoding='utf-8', errors='replace'):
            m = TS.match(line.strip())
            if not m: continue
            raw = m.group(10).strip(); c = clean_human(raw)
            out[clip].append(dict(st=sec(*m.group(1, 2, 3)), en=sec(*m.group(4, 5, 6)), lang=m.group(7),
                                  spk=m.group(8), raw=raw, c=c, lex=bool(c)))
    return out

def overlap(a, b): return min(a['en'], b['en']) - max(a['st'], b['st'])

# ------------------------------------------------------------------ analyses
def stock_and_labels(pipe, hum, summary):
    allseg = [s for c in pipe for s in pipe[c]]
    # language-label agreement (Results, "Language labeling")
    words = agree = 0
    for s in allseg:
        if s['text'] and s['tag'] in ('en', 'es'):
            k = len(s['text'].split()); words += k; agree += k * (s['tag'] == s['wl'])
    summary['label_words_en_es'] = words; summary['label_words_agree'] = agree
    summary['label_agreement_pct'] = round(100 * agree / words, 2)

    eng = [s for s in allseg if s['tag'] == 'en' and s['n']]
    cnt = Counter(s['n'] for s in eng)
    top = [p for p, _ in cnt.most_common(25)]
    summary['english_utterances'] = len(eng)
    summary['top25_utterances'] = sum(cnt[p] for p in top)
    summary['top25_pct'] = round(100 * summary['top25_utterances'] / len(eng), 1)
    summary['top3_utterances'] = cnt['thank you'] + cnt['yeah'] + cnt['you']

    def status(clip, s):
        ov = [u for u in hum.get(clip, []) if overlap(s, u) > 0]
        if not ov: return 'nothing'
        return 'speech' if any(u['lex'] for u in ov) else 'nonlexical'

    hcnt = Counter(' '.join(clean_text(u['raw']).split()) for c in hum for u in hum[c] if clean_text(u['raw']))
    rows = []
    for rank, p in enumerate(top, 1):
        inst = [(c, s) for c in pipe for s in pipe[c] if s['tag'] == 'en' and s['n'] == p]
        sc = Counter(status(c, s) for c, s in inst)
        rows.append(dict(rank=rank, utterance=p, pipeline=len(inst), coders=hcnt[p.replace("'", '')],
                         nothing=sc['nothing'], nonlexical=sc['nonlexical'], speech=sc['speech'],
                         non_speech_pct=round(100 * (sc['nothing'] + sc['nonlexical']) / len(inst), 1)))
    stock = [(c, s) for c in pipe for s in pipe[c] if s['tag'] == 'en' and s['n'] in set(top)]
    sc = Counter(status(c, s) for c, s in stock); n = len(stock)
    summary['stock_nothing_pct'] = round(100 * sc['nothing'] / n, 2)
    summary['stock_nonlexical_pct'] = round(100 * sc['nonlexical'] / n, 2)
    summary['stock_nonspeech_pct'] = round(100 * (sc['nothing'] + sc['nonlexical']) / n, 2)
    lang = Counter()
    for c, s in stock:
        ov = [u for u in hum.get(c, []) if overlap(s, u) > 0 and u['lex']]
        if ov: lang[max(ov, key=lambda u: overlap(s, u))['lang']] += 1
    k = sum(lang.values())
    summary['stock_overlap_speech_segments'] = k
    summary['stock_overlap_spanish_pct'] = round(100 * lang['es'] / k, 1)
    summary['stock_overlap_english_pct'] = round(100 * lang['en'] / k, 1)
    ew = sum(len(s['text'].split()) for s in eng)
    ewn = sum(len(s['text'].split()) for c in pipe for s in pipe[c]
              if s['tag'] == 'en' and s['n'] and status(c, s) != 'speech')
    summary['english_words'] = ew; summary['english_words_nonspeech'] = ewn
    summary['english_words_nonspeech_pct'] = round(100 * ewn / ew, 1)

    # language-neutral interjections in pipeline-English utterances
    neut = {'hm', 'hmm', 'mm', 'mhm', 'okay', 'ok', 'no', 'yeah', 'oh', 'uh', 'um', 'ah', 'eh'}
    pe = Counter(w for s in eng for w in s['n'].split() if w in neut)
    tot = sum(len(s['n'].split()) for s in eng)
    summary['neutral_words_in_english'] = sum(pe.values())
    summary['neutral_words_in_english_excl_no'] = sum(pe.values()) - pe['no']
    summary['words_in_english_utterances'] = tot

    # by speaker
    by = defaultdict(Counter)
    for s in allseg:
        by[s['spk']]['all_segments'] += 1
        if not s['text']: by[s['spk']]['empty_segments'] += 1
    for s in eng: by[s['spk']]['english_utterances'] += 1
    for _, s in stock: by[s['spk']]['stock_segments'] += 1
    emp = [s for s in allseg if not s['text']]
    summary['empty_segments'] = len(emp)
    summary['empty_minutes'] = round(sum(s['en'] - s['st'] for s in emp) / 60, 1)
    summary['empty_overlapping'] = sum(s['ov'] for s in emp)
    return rows, by, emp

def discarded(pipe, nolang, summary):
    nl = {(c, s['key']): s for c in nolang for s in nolang[c]}
    emp = [(c, s) for c in pipe for s in pipe[c] if not s['text']]
    matched = [(c, s) for c, s in emp if (c, s['key']) in nl]
    other = [nl[(c, s['key'])] for c, s in matched if nl[(c, s['key'])]['text'] and nl[(c, s['key'])]['wl'] not in ('en', 'es')]
    summary['empty_matched_in_nolang'] = len(matched)
    summary['empty_with_other_language_text'] = len(other)
    summary['empty_other_language_words'] = sum(len(s['text'].split()) for s in other)
    langs = Counter(s['wl'] for s in other); words = Counter()
    for s in other: words[s['wl']] += len(s['text'].split())
    return [dict(language=l, segments=n, words=words[l]) for l, n in langs.most_common()]

def unassigned(root, label, summary):
    line = re.compile(r'^\[[^\]]+\]\s*\(\w+\)\s*\[([A-Z]+)\](?:\s*\[OVERLAP\])?:\s*(.*)$')
    c = Counter()
    for f in glob.glob(os.path.join(root, '**', '*_tagged.txt'), recursive=True):
        m = re.search(r'_(KCHI|OCH|FEM|MAL|UNKNOWN|SPEECH)_tagged\.txt$', f)
        if not m: continue
        for l in open(f, encoding='utf-8', errors='replace'):
            mm = line.match(l.strip())
            if mm: c[m.group(1)] += len(mm.group(2).split())
    tot = sum(c.values()); un = c['UNKNOWN'] + c['SPEECH']
    summary[f'{label}_words'] = tot; summary[f'{label}_unassigned_words'] = un
    summary[f'{label}_unassigned_pct'] = round(100 * un / tot, 1)

def segments(v1root, v2root, hum, summary):
    def load_vtc(root):
        out = defaultdict(list)
        for f in glob.glob(os.path.join(root, '**', '*.txt'), recursive=True):
            m = re.search(r'^(.*_\d{3})_([A-Z]+)\.txt$', os.path.basename(f))
            if not m: continue
            for l in open(f, encoding='utf-8', errors='replace'):
                mm = SEGLINE.match(l.strip())
                if mm: out[ck(m.group(1))].append((sec(*mm.group(1, 2, 3)), sec(*mm.group(4, 5, 6)), m.group(2)))
        return out
    def merged(iv):
        out = []
        for s, e in sorted(iv):
            if out and s <= out[-1][1]: out[-1][1] = max(out[-1][1], e)
            else: out.append([s, e])
        return out
    def inter(a, b):
        i = j = 0; t = 0.0
        while i < len(a) and j < len(b):
            t += max(0.0, min(a[i][1], b[j][1]) - max(a[i][0], b[j][0]))
            if a[i][1] < b[j][1]: i += 1
            else: j += 1
        return t
    V = {'vtc1': load_vtc(v1root), 'vtc2': load_vtc(v2root)}
    H = {c: [(u['st'], u['en'], u['spk']) for u in us] for c, us in hum.items()}
    clips = sorted(set(V['vtc1']) & set(V['vtc2']) & set(H))
    summary['segment_clips'] = len(clips)
    AD = ('FEM', 'MAL')
    coder_adult = sum(e - s for c in clips for s, e in merged([(s, e) for s, e, l in H[c] if l in AD]))
    summary['coder_adult_minutes'] = round(coder_adult / 60, 1)
    for k, segs in V.items():
        d = [e - s for c in clips for s, e, l in segs[c] if l in AD]
        summary[f'{k}_adult_segment_median_s'] = round(st.median(d), 2)
        split = n = 0
        for c in clips:
            a = [x for x in segs[c] if x[2] in AD]
            for s, e, l in H[c]:
                if l in AD:
                    n += 1; split += sum(min(e, y[1]) - max(s, y[0]) > 0 for y in a) >= 2
        summary[f'{k}_adult_utterances_split_pct'] = round(100 * split / n, 1)
        tot = ina = inany = 0.0
        for c in clips:
            A = merged([(s, e) for s, e, l in segs[c] if l in AD])
            tot += sum(e - s for s, e in A)
            ina += inter(A, merged([(s, e) for s, e, l in H[c] if l in AD]))
            inany += inter(A, merged([(s, e) for s, e, l in H[c]]))
        summary[f'{k}_adult_minutes'] = round(tot / 60, 1)
        summary[f'{k}_adult_time_in_coder_adult_pct'] = round(100 * ina / tot, 1)
        summary[f'{k}_adult_time_on_coder_child_pct'] = round(100 * (inany - ina) / tot, 1)
        summary[f'{k}_adult_time_uncoded_pct'] = round(100 * (tot - inany) / tot, 1)

def reference(wc_dir, clean_dir, human_dir, summary):
    tot = Counter(); n_xxx = n_words = 0
    for f in glob.glob(os.path.join(wc_dir, '*', 'combined_statistics.csv')):
        for r in csv.DictReader(open(f, encoding='utf-8-sig')):
            if r['Audio_File'].endswith('COMBINED'):
                for k in ('Total_Words', 'Total_English_Words', 'Total_Spanish_Words'): tot[k] += int(float(r[k]))
    for f in glob.glob(os.path.join(wc_dir, '*', '*_words.txt')):
        t = open(f, encoding='utf-8', errors='replace').read()
        m = re.search(r'All \w+ words \(in order of appearance\):\s*\n(.*?)(?:\n\s*\n[A-Z]{3,}\s|\Z)', t, flags=re.S)
        if m: n_xxx += len(re.findall(r'\bxxx\b', m.group(1), flags=re.I)); n_words += len(m.group(1).split())
    ref = sum(len(open(f, encoding='utf-8', errors='replace').read().split()) for f in overall_files(clean_dir, '_clean.txt')
              if not re.search(r'_(ADULT|CHILDREN)_clean\.txt$', f))
    summary.update(reference_words=tot['Total_Words'], reference_english=tot['Total_English_Words'],
                   reference_spanish=tot['Total_Spanish_Words'], reference_xxx_tokens=n_xxx,
                   wer_reference_words=ref, reference_gap=tot['Total_Words'] - ref)
    by_tag = Counter()
    for f in glob.glob(os.path.join(human_dir, '*', '*', '*.txt')):
        for line in open(f, encoding='utf-8', errors='replace'):
            m = TS.match(line.strip())
            if m: by_tag[m.group(7)] += len(clean_text(m.group(10)).split())
    summary['reference_mixed_words'] = by_tag['both']
    summary['reference_unmarked_words'] = sum(v for k, v in by_tag.items() if k not in ('en', 'es', 'both'))

# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for a in ('vtc2-langdetect', 'nolang-langdetect', 'human-dir', 'posthoc-vtc1', 'posthoc-vtc2',
              'vtc1-seg', 'vtc2-seg', 'human-wordcount', 'human-clean'):
        ap.add_argument('--' + a, required=True)
    ap.add_argument('--out', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'derived', 'failure_analysis'))
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    summary = {}
    pipe, hum = load_pipeline(a.vtc2_langdetect), load_human(a.human_dir)
    rows, by, _ = stock_and_labels(pipe, hum, summary)
    langs = discarded(pipe, load_pipeline(a.nolang_langdetect), summary)
    unassigned(a.posthoc_vtc1, 'posthoc_vtc1', summary); unassigned(a.posthoc_vtc2, 'posthoc_vtc2', summary)
    segments(a.vtc1_seg, a.vtc2_seg, hum, summary)
    reference(a.human_wordcount, a.human_clean, a.human_dir, summary)

    def wcsv(name, rows, fields):
        with open(os.path.join(a.out, name), 'w', newline='', encoding='utf-8') as fh:
            w = csv.DictWriter(fh, fieldnames=fields); w.writeheader(); w.writerows(rows)
        print('wrote', os.path.join(a.out, name))
    wcsv('summary.csv', [dict(metric=k, value=v) for k, v in summary.items()], ['metric', 'value'])
    wcsv('stock_phrases.csv', rows, list(rows[0]))
    wcsv('discarded_languages.csv', langs, ['language', 'segments', 'words'])
    wcsv('segments_by_speaker.csv', [dict(speaker=k, **{f: v[f] for f in ('all_segments', 'empty_segments', 'english_utterances', 'stock_segments')})
                                     for k, v in sorted(by.items())],
         ['speaker', 'all_segments', 'empty_segments', 'english_utterances', 'stock_segments'])

if __name__ == '__main__':
    main()
