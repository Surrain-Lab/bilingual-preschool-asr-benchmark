import re
import os
import unicodedata
from datetime import datetime
import jiwer
from jiwer import cer
def compute_measures(reference, hypothesis):
    r = jiwer.process_words(reference, hypothesis)
    return {'wer': r.wer, 'mer': r.mer, 'wil': r.wil, 'wip': r.wip,
            'hits': r.hits, 'substitutions': r.substitutions,
            'deletions': r.deletions, 'insertions': r.insertions}
import difflib
import string
import argparse
from langdetect import detect, DetectorFactory, LangDetectException
from collections import Counter
DetectorFactory.seed = 0

def read_text_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def remove_special_annotations(text):
    text = re.sub(r'\[.*?\]', '', text)
    text = text.replace('+//', '').replace('+<', '')
    text = re.sub(r'\bxxx\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'[@&][^\s]+', '', text)
    text = text.replace('+', '').replace('-', '')
    text = re.sub(r'\bde el\b', 'del', text, flags=re.IGNORECASE)
    text = re.sub(r'\ba el\b', 'al',  text, flags=re.IGNORECASE)
    return text

def remove_diacritics(text):
    nfd = unicodedata.normalize('NFD', text)
    return ''.join(c for c in nfd if unicodedata.category(c) != 'Mn')

def remove_punctuation_and_special_chars(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

def _detect_lang_chunk(text, lang_hint=None):
    """Detect 'es' or 'en' for a chunk via langdetect with consensus.
    Mirrors the langdetect_tags.py strategy."""
    text = text.strip()
    if not text:
        return 'unknown'
    text_lower = text.lower()
    if any(c in text_lower for c in 'áéíóúñü¿¡'):
        return 'es'
    word_count = len(text.split())
    results = []           # all langdetect results
    es_en_results = []     # only es/en detections
    for _ in range(15):
        try:
            d = detect(text)
            results.append(d)
            if d in ('en', 'es'):
                es_en_results.append(d)
        except LangDetectException:
            pass
    # Bilingual EN/ES benchmark logic:
    # 1. If any en/es detections, trust the majority among them.
    #    This handles short Spanish chunks where langdetect may also
    #    occasionally return pt/ca/gl by chance.
    if es_en_results:
        counts = Counter(es_en_results)
        top, top_n = counts.most_common(1)[0]
        confidence = top_n / len(es_en_results)
        threshold = 0.5 if word_count <= 3 else 0.4
        if confidence >= threshold:
            return top
        return top
    # 2. langdetect detected something but never en/es. Two cases:
    #    a) Short text (<= 5 words) — too ambiguous to trust langdetect's
    #       non-en/es picks; fall back to lang_hint (typically 'es' here).
    #    b) Longer text — langdetect was reasonably confident it is not
    #       Spanish, so call it English (bilingual benchmark default).
    if results:
        if word_count <= 5 and lang_hint in ('en', 'es'):
            return lang_hint
        return 'en'
    # 3. No detection at all — fall back to the language hint.
    if lang_hint in ('en', 'es'):
        return lang_hint
    return 'unknown'


def analyze_language_distribution(text, lang_hint='es'):
    """Per-utterance language detection on raw text, then word counts per lang.
    Pass *raw* text (with diacritics) so the Spanish-character heuristic works."""
    if not text or not text.strip():
        return {'spanish_words': 0, 'english_words': 0, 'total_words': 0,
                'spanish_percentage': 0, 'english_percentage': 0}

    chunks = re.split(r'(?<=[.?!])\s+|\n+', text)
    chunks = [c.strip() for c in chunks if c and c.strip()]
    if not chunks:
        chunks = [text.strip()]

    spanish_count = 0
    english_count = 0
    for chunk in chunks:
        words = re.findall(r"[A-Za-zÁÉÍÓÚáéíóúÑñÜü]+", chunk)
        n = len(words)
        if n == 0:
            continue
        lang = _detect_lang_chunk(chunk, lang_hint=lang_hint)
        if lang == 'en':
            english_count += n
        elif lang == 'es':
            spanish_count += n
        else:
            if lang_hint == 'en':
                english_count += n
            else:
                spanish_count += n

    total = spanish_count + english_count
    return {
        'spanish_words':      spanish_count,
        'english_words':      english_count,
        'total_words':        total,
        'spanish_percentage': (spanish_count / total * 100) if total > 0 else 0,
        'english_percentage': (english_count / total * 100) if total > 0 else 0,
    }

def clean_text(text):
    text = remove_special_annotations(text)
    text = re.sub(r'[¿¡]', '', text)
    text = remove_diacritics(text)
    text = remove_punctuation_and_special_chars(text)
    text = ' '.join(text.split()).lower()
    return text

def calculate_detailed_metrics(reference, hypothesis):
    measures       = compute_measures(reference, hypothesis)
    char_error_rate = cer(reference, hypothesis)
    return {
        'wer':             measures['wer'],
        'mer':             measures['mer'],
        'wil':             measures['wil'],
        'wip':             measures['wip'],
        'cer':             char_error_rate,
        'hits':            measures['hits'],
        'substitutions':   measures['substitutions'],
        'deletions':       measures['deletions'],
        'insertions':      measures['insertions'],
        'ref_words_count': len(reference.split()),
        'hyp_words_count': len(hypothesis.split()),
        'ref_chars_count': len(reference),
        'hyp_chars_count': len(hypothesis),
    }

def get_alignment_details(reference, hypothesis):
    details = {'hits': [], 'substitutions': [], 'deletions': [], 'insertions': []}

    output     = jiwer.process_words(reference, hypothesis)
    ref_words  = output.references[0]
    hyp_words  = output.hypotheses[0]
    alignments = output.alignments[0]
    for chunk in alignments:
        r_words = ref_words[chunk.ref_start_idx:chunk.ref_end_idx]
        h_words = hyp_words[chunk.hyp_start_idx:chunk.hyp_end_idx]
        if chunk.type == 'equal':
            details['hits'].extend(r_words)
        elif chunk.type == 'substitute':
            minlen = min(len(r_words), len(h_words))
            for rw, hw in zip(r_words[:minlen], h_words[:minlen]):
                details['substitutions'].append((rw, hw))
            details['deletions'].extend(r_words[minlen:])
            details['insertions'].extend(h_words[minlen:])
        elif chunk.type == 'delete':
            details['deletions'].extend(r_words)
        elif chunk.type == 'insert':
            details['insertions'].extend(h_words)
    return details

def format_detailed_report(language, speaker, results, alignment_details,
                            ref_text, hyp_text, ref_lang_dist, hyp_lang_dist, timestamp):
    lines = []
    lines.append("=" * 80)
    lines.append(f"DETAILED TRANSCRIPTION EVALUATION REPORT - {language.upper()}")
    lines.append("=" * 80)
    lines.append(f"Generated on: {timestamp}")
    lines.append(f"Speaker: {speaker}")
    lines.append(f"Language: {language}")
    lines.append("")
    lines.append("Preprocessing Applied:")
    lines.append("  - CHAT format annotations removed")
    lines.append("  - Special characters and punctuation removed")
    lines.append("  - All diacritics removed (including n -> n)")
    lines.append("  - Spanish contractions normalized (de el -> del, a el -> al)")
    lines.append("  - Inverted Spanish punctuation removed")
    lines.append("  - Text converted to lowercase")
    lines.append("")
    lines.append("-" * 80)
    lines.append("LANGUAGE DISTRIBUTION ANALYSIS")
    lines.append("-" * 80)
    lines.append("Reference (Ground Truth):")
    lines.append(f"  Spanish words: {ref_lang_dist['spanish_words']} ({ref_lang_dist['spanish_percentage']:.2f}%)")
    lines.append(f"  English words: {ref_lang_dist['english_words']} ({ref_lang_dist['english_percentage']:.2f}%)")
    lines.append(f"  Total words:   {ref_lang_dist['total_words']}")
    lines.append("")
    lines.append("Hypothesis (Trainee):")
    lines.append(f"  Spanish words: {hyp_lang_dist['spanish_words']} ({hyp_lang_dist['spanish_percentage']:.2f}%)")
    lines.append(f"  English words: {hyp_lang_dist['english_words']} ({hyp_lang_dist['english_percentage']:.2f}%)")
    lines.append(f"  Total words:   {hyp_lang_dist['total_words']}")
    lines.append("")
    lines.append("-" * 80)
    lines.append("WORD-LEVEL ANALYSIS")
    lines.append("-" * 80)
    lines.append(f"Word Error Rate (WER):            {results['wer']:.4f} ({results['wer']:.2%})")
    lines.append(f"Match Error Rate (MER):           {results['mer']:.4f} ({results['mer']:.2%})")
    lines.append(f"Word Information Lost (WIL):      {results['wil']:.4f} ({results['wil']:.2%})")
    lines.append(f"Word Information Preserved (WIP): {results['wip']:.4f} ({results['wip']:.2%})")
    lines.append("")
    lines.append("Word Counts:")
    lines.append(f"  Total Reference Words:  {results['ref_words_count']}")
    lines.append(f"  Total Hypothesis Words: {results['hyp_words_count']}")
    lines.append(f"  Hits (Correct):         {results['hits']}")
    lines.append(f"  Substitutions:          {results['substitutions']}")
    lines.append(f"  Deletions:              {results['deletions']}")
    lines.append(f"  Insertions:             {results['insertions']}")
    lines.append(f"  Total Errors:           {results['substitutions'] + results['deletions'] + results['insertions']}")
    lines.append(f"  Accuracy:               {(1 - results['wer']) * 100:.2f}%")
    lines.append("")

    if alignment_details['hits']:
        lines.append(f"Correctly Recognized Words - Total: {len(alignment_details['hits'])}:")
        for i, w in enumerate(alignment_details['hits'][:50], 1):
            lines.append(f"  {i}. '{w}'")
        if len(alignment_details['hits']) > 50:
            lines.append(f"  ... and {len(alignment_details['hits']) - 50} more")
        lines.append("")

    if alignment_details['substitutions']:
        lines.append(f"Substitutions (Reference -> Hypothesis) - Total: {len(alignment_details['substitutions'])}:")
        for i, (rw, hw) in enumerate(alignment_details['substitutions'], 1):
            lines.append(f"  {i}. '{rw}' -> '{hw}'")
        lines.append("")

    if alignment_details['deletions']:
        lines.append(f"Deletions (Missing from hypothesis) - Total: {len(alignment_details['deletions'])}:")
        for i, w in enumerate(alignment_details['deletions'], 1):
            lines.append(f"  {i}. '{w}'")
        lines.append("")

    if alignment_details['insertions']:
        lines.append(f"Insertions (Extra in hypothesis) - Total: {len(alignment_details['insertions'])}:")
        for i, w in enumerate(alignment_details['insertions'], 1):
            lines.append(f"  {i}. '{w}'")
        lines.append("")

    lines.append("-" * 80)
    lines.append("CHARACTER-LEVEL ANALYSIS")
    lines.append("-" * 80)
    lines.append(f"Character Error Rate (CER):  {results['cer']:.4f} ({results['cer']:.2%})")
    lines.append(f"Total Reference Characters:  {results['ref_chars_count']}")
    lines.append(f"Total Hypothesis Characters: {results['hyp_chars_count']}")
    lines.append("")
    lines.append("-" * 80)
    lines.append("COMPLETE TEXT")
    lines.append("-" * 80)
    lines.append(f"Reference - Total length: {len(ref_text)} characters:")
    lines.append(f"  {ref_text[:500]}{'...' if len(ref_text) > 500 else ''}")
    lines.append("")
    lines.append(f"Hypothesis - Total length: {len(hyp_text)} characters:")
    lines.append(f"  {hyp_text[:500]}{'...' if len(hyp_text) > 500 else ''}")
    lines.append("")
    return lines

def write_detailed_report(result, filename):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = format_detailed_report(
        result['language'], result['speaker'], result['results'],
        result['alignment_details'], result['ref_text'], result['hyp_text'],
        result['ref_lang_dist'], result['hyp_lang_dist'], timestamp
    )
    with open(filename, 'w', encoding='utf-8') as f:
        for line in lines:
            f.write(line + '\n')
    print(f"  Report saved: {filename}")

def generate_html_report(result, html_filename):
    now               = datetime.now()
    timestamp_display = now.strftime("%Y-%m-%d %H:%M:%S")
    wer      = result['results']['wer'] * 100
    cer_val  = result['results']['cer'] * 100
    wip      = result['results']['wip'] * 100
    accuracy = (1 - result['results']['wer']) * 100
    ref_lang = result['ref_lang_dist']
    hyp_lang = result['hyp_lang_dist']
    accuracy_class = 'good' if accuracy >= 90 else 'warning' if accuracy >= 70 else 'error'
    wer_class      = 'good' if wer < 10      else 'warning' if wer < 30      else 'error'
    cer_class      = 'good' if cer_val < 10  else 'warning' if cer_val < 30  else 'error'

    substitutions_html = ''.join(
        f'<div class="error-item substitution"><strong>{i+1}.</strong> {rw} → {hw}</div>'
        for i, (rw, hw) in enumerate(result['alignment_details']['substitutions'])
    )

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Transcript Evaluation - {result['speaker']} ({result['language']})</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/3.9.1/chart.min.js"></script>
    <style>
        * {{ margin:0; padding:0; box-sizing:border-box; }}
        body {{ font-family:'Segoe UI',sans-serif; background:linear-gradient(135deg,#667eea,#764ba2); min-height:100vh; padding:20px; }}
        .container {{ max-width:1400px; margin:0 auto; background:white; border-radius:20px; box-shadow:0 20px 60px rgba(0,0,0,.3); overflow:hidden; }}
        .header {{ background:linear-gradient(135deg,#667eea,#764ba2); color:white; padding:40px; text-align:center; }}
        .header h1 {{ font-size:2.5em; margin-bottom:10px; }}
        .content {{ padding:40px; }}
        .info-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr)); gap:20px; margin-bottom:30px; }}
        .info-card {{ background:linear-gradient(135deg,#f5f7fa,#c3cfe2); padding:20px; border-radius:15px; }}
        .info-card h3 {{ color:#667eea; font-size:.9em; text-transform:uppercase; margin-bottom:10px; }}
        .metrics-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:20px; margin:30px 0; }}
        .metric-card {{ background:white; border-left:4px solid #667eea; padding:20px; border-radius:10px; box-shadow:0 2px 8px rgba(0,0,0,.1); transition:all .3s; }}
        .metric-card:hover {{ transform:translateY(-5px); box-shadow:0 5px 15px rgba(0,0,0,.2); }}
        .metric-card.good {{ border-left-color:#48bb78; }}
        .metric-card.warning {{ border-left-color:#ed8936; }}
        .metric-card.error {{ border-left-color:#f56565; }}
        .metric-value {{ font-size:2em; font-weight:bold; color:#2d3748; }}
        .metric-label {{ font-size:.85em; color:#718096; margin-bottom:5px; }}
        .section {{ margin:40px 0; background:#f8f9fa; padding:30px; border-radius:15px; }}
        .section h2 {{ color:#667eea; margin-bottom:20px; font-size:1.8em; border-bottom:3px solid #667eea; padding-bottom:10px; }}
        .chart-container {{ background:white; padding:20px; border-radius:10px; margin:20px 0; }}
        .lang-stats {{ display:grid; grid-template-columns:1fr 1fr; gap:20px; margin:20px 0; }}
        .lang-card {{ background:white; padding:20px; border-radius:10px; }}
        .lang-card h3 {{ color:#667eea; margin-bottom:15px; }}
        .error-list {{ background:white; border-radius:10px; padding:20px; max-height:400px; overflow-y:auto; }}
        .error-item {{ padding:10px; margin:6px 0; border-radius:8px; font-family:'Courier New',monospace; font-size:.9em; }}
        .error-item.substitution {{ background:#fff5f5; border-left:4px solid #f56565; }}
        .collapsible {{ cursor:pointer; padding:15px; background:#667eea; color:white; border:none; border-radius:8px; width:100%; text-align:left; font-size:1.1em; margin:10px 0; }}
        .collapsible:hover {{ background:#764ba2; }}
        .collapsible:after {{ content:'+'; float:right; }}
        .collapsible.active:after {{ content:'-'; }}
        .collapsible-content {{ max-height:0; overflow:hidden; transition:max-height .3s ease-out; }}
        .collapsible-content.active {{ max-height:2000px; overflow-y:auto; }}
    </style>
</head>
<body>
<div class="container">
    <div class="header">
        <h1>Transcript Evaluation Report</h1>
        <p>{result['language']} — Speaker: {result['speaker']}</p>
    </div>
    <div class="content">
        <div class="info-grid">
            <div class="info-card"><h3>Speaker</h3><p>{result['speaker']}</p></div>
            <div class="info-card"><h3>Language</h3><p>{result['language']}</p></div>
            <div class="info-card"><h3>Report Date</h3><p>{timestamp_display}</p></div>
            <div class="info-card"><h3>Total Ref Words</h3><p>{result['results']['ref_words_count']}</p></div>
        </div>

        <div class="section">
            <h2>Language Distribution</h2>
            <div class="lang-stats">
                <div class="lang-card">
                    <h3>Reference (Ground Truth)</h3>
                    <p>Spanish: {ref_lang['spanish_words']} words ({ref_lang['spanish_percentage']:.2f}%)</p>
                    <p>English: {ref_lang['english_words']} words ({ref_lang['english_percentage']:.2f}%)</p>
                    <p>Total: {ref_lang['total_words']} words</p>
                </div>
                <div class="lang-card">
                    <h3>Hypothesis (Trainee)</h3>
                    <p>Spanish: {hyp_lang['spanish_words']} words ({hyp_lang['spanish_percentage']:.2f}%)</p>
                    <p>English: {hyp_lang['english_words']} words ({hyp_lang['english_percentage']:.2f}%)</p>
                    <p>Total: {hyp_lang['total_words']} words</p>
                </div>
            </div>
            <div class="chart-container"><canvas id="langChart"></canvas></div>
        </div>

        <div class="metrics-grid">
            <div class="metric-card {accuracy_class}"><div class="metric-label">Accuracy</div><div class="metric-value">{accuracy:.2f}%</div></div>
            <div class="metric-card {wer_class}"><div class="metric-label">Word Error Rate</div><div class="metric-value">{wer:.2f}%</div></div>
            <div class="metric-card {cer_class}"><div class="metric-label">Char Error Rate</div><div class="metric-value">{cer_val:.2f}%</div></div>
            <div class="metric-card"><div class="metric-label">Word Info Preserved</div><div class="metric-value">{wip:.2f}%</div></div>
        </div>

        <div class="section">
            <h2>Error Distribution</h2>
            <div class="chart-container"><canvas id="errorChart"></canvas></div>
        </div>

        <div class="section">
            <h2>Substitutions ({len(result['alignment_details']['substitutions'])})</h2>
            <button class="collapsible">View All Substitutions</button>
            <div class="collapsible-content">
                <div class="error-list">{substitutions_html}</div>
            </div>
        </div>
    </div>
</div>
<script>
    document.querySelectorAll('.collapsible').forEach(c => {{
        c.addEventListener('click', function() {{
            this.classList.toggle('active');
            this.nextElementSibling.classList.toggle('active');
        }});
    }});

    new Chart(document.getElementById('langChart').getContext('2d'), {{
        type: 'bar',
        data: {{
            labels: ['Reference', 'Hypothesis'],
            datasets: [
                {{ label: 'Spanish', data: [{ref_lang['spanish_words']}, {hyp_lang['spanish_words']}], backgroundColor: '#f56565' }},
                {{ label: 'English', data: [{ref_lang['english_words']}, {hyp_lang['english_words']}], backgroundColor: '#4299e1' }}
            ]
        }},
        options: {{ responsive:true, plugins:{{ legend:{{ position:'bottom' }}, title:{{ display:true, text:'Spanish vs English Word Distribution' }} }}, scales:{{ x:{{ stacked:true }}, y:{{ stacked:true }} }} }}
    }});

    new Chart(document.getElementById('errorChart').getContext('2d'), {{
        type: 'doughnut',
        data: {{
            labels: ['Correct','Substitutions','Deletions','Insertions'],
            datasets: [{{ data:[{result['results']['hits']},{result['results']['substitutions']},{result['results']['deletions']},{result['results']['insertions']}], backgroundColor:['#48bb78','#f56565','#ed8936','#4299e1'] }}]
        }},
        options: {{ responsive:true, plugins:{{ legend:{{ position:'bottom' }}, title:{{ display:true, text:'Word-Level Error Distribution' }} }} }}
    }});
</script>
</body>
</html>"""

    with open(html_filename, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"  HTML saved: {html_filename}")

def process_speaker(ground_truth_file, whisperx_file, speaker, language):
    print(f"\n  [{speaker}]")
    ground_truth  = read_text_file(ground_truth_file)
    whisperx_text = read_text_file(whisperx_file)

    if not ground_truth or not whisperx_text:
        print(f"  Error: Could not read files for {speaker}")
        return None

    gt_clean  = clean_text(ground_truth)
    wx_clean  = clean_text(whisperx_text)
    # Use RAW text for language detection so Spanish characters survive.
    _hint = 'es' if language.lower().startswith('span') else 'en'
    ref_lang  = analyze_language_distribution(ground_truth, lang_hint=_hint)
    hyp_lang  = analyze_language_distribution(whisperx_text, lang_hint=_hint)
    results   = calculate_detailed_metrics(gt_clean, wx_clean)
    alignment = get_alignment_details(gt_clean, wx_clean)

    print(f"  WER: {results['wer']:.2%}  CER: {results['cer']:.2%}  Accuracy: {(1-results['wer'])*100:.2f}%")

    return {
        'speaker':          speaker,
        'language':         language,
        'results':          results,
        'alignment_details': alignment,
        'ref_text':         gt_clean,
        'hyp_text':         wx_clean,
        'ref_lang_dist':    ref_lang,
        'hyp_lang_dist':    hyp_lang,
    }

def write_combined_report(all_results, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 120 + "\n")
        f.write("COMBINED TRANSCRIPTION EVALUATION REPORT - ALL SPEAKERS\n")
        f.write("=" * 120 + "\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        speakers = list(all_results.keys())

        # WER metrics table
        f.write("-" * 120 + "\n")
        f.write("SUMMARY COMPARISON - WER METRICS\n")
        f.write("-" * 120 + "\n")
        header = f"{'Metric':<30} " + "".join(f"{s:<18}" for s in speakers) + "\n"
        f.write(header)
        f.write("-" * 120 + "\n")
        for metric in ['wer', 'mer', 'wil', 'wip', 'cer']:
            line = f"{metric.upper():<30} " + "".join(f"{all_results[s]['results'][metric]:<18.4f}" for s in speakers) + "\n"
            f.write(line)

        # Word counts
        f.write("\n" + "-" * 120 + "\n")
        f.write("WORD COUNTS AND ACCURACY\n")
        f.write("-" * 120 + "\n")
        for ct in ['ref_words_count', 'hits', 'substitutions', 'insertions', 'deletions']:
            line = f"{ct:<30} " + "".join(f"{all_results[s]['results'][ct]:<18}" for s in speakers) + "\n"
            f.write(line)
        acc_line = f"{'Accuracy %':<30} " + "".join(f"{(1-all_results[s]['results']['wer'])*100:<17.2f}%" for s in speakers) + "\n"
        f.write(acc_line)

        # Language distribution — reference
        f.write("\n" + "=" * 120 + "\n")
        f.write("LANGUAGE DISTRIBUTION - REFERENCE (GROUND TRUTH)\n")
        f.write("=" * 120 + "\n")
        hdr = f"{'Speaker':<30} {'Spanish Words':<18} {'English Words':<18} {'Total':<18} {'Spanish %':<15}\n"
        f.write(hdr)
        f.write("-" * 120 + "\n")
        for s in speakers:
            ld = all_results[s]['ref_lang_dist']
            f.write(f"{s:<30} {ld['spanish_words']:<18} {ld['english_words']:<18} {ld['total_words']:<18} {ld['spanish_percentage']:<14.2f}%\n")

        # Language distribution — hypothesis
        f.write("\n" + "=" * 120 + "\n")
        f.write("LANGUAGE DISTRIBUTION - HYPOTHESIS (TRAINEE)\n")
        f.write("=" * 120 + "\n")
        f.write(hdr)
        f.write("-" * 120 + "\n")
        for s in speakers:
            ld = all_results[s]['hyp_lang_dist']
            f.write(f"{s:<30} {ld['spanish_words']:<18} {ld['english_words']:<18} {ld['total_words']:<18} {ld['spanish_percentage']:<14.2f}%\n")

        # Best performer
        f.write("\n" + "=" * 120 + "\n")
        best = min(speakers, key=lambda s: all_results[s]['results']['wer'])
        f.write(f"BEST PERFORMER: {best}\n")
        f.write(f"WER: {all_results[best]['results']['wer']:.4f}\n")
        f.write("=" * 120 + "\n")

    print(f"  Combined report: {output_file}")

def process_single_file(base_filename, clip_name, gt_base_dir, wx_base_dir, output_base_dir, language='English'):
    output_dir = os.path.join(output_base_dir, base_filename, clip_name)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"Recording: {base_filename}  Clip: {clip_name}")
    print(f"Output:    {output_dir}")
    print("=" * 60)

    # ── Speaker configurations ──────────────────────────────────────────
    # (display_name, file_suffix)
    # OVERALL   -> clip_name_clean.txt          (MAL+FEM+KCHI+OCH combined)
    # ADULT     -> clip_name_ADULT_clean.txt     (MAL+FEM)
    # CHILDREN  -> clip_name_CHILDREN_clean.txt  (KCHI+OCH)   ← NEW
    # KCHI      -> clip_name_KCHI_clean.txt
    # OCH       -> clip_name_OCH_clean.txt       ← NEW
    # MAL       -> clip_name_MAL_clean.txt
    # FEM       -> clip_name_FEM_clean.txt
    # ───────────────────────────────────────────────────────────────────
    speaker_configs = [
        ('MAL',      'MAL'),
        ('FEM',      'FEM'),
        ('KCHI',     'KCHI'),
        ('OCH',      'OCH'),          # Other child
        ('ADULT',    'ADULT'),        # MAL + FEM combined
        ('CHILDREN', 'CHILDREN'),     # KCHI + OCH combined
        ('OVERALL',  ''),             # everyone except UNKNOWN/SPEECH
    ]

    all_results = {}

    for speaker_display, suffix in speaker_configs:
        if speaker_display == 'OVERALL':
            gt_file = os.path.join(gt_base_dir, base_filename, clip_name, f"{clip_name}_clean.txt")
            wx_file = os.path.join(wx_base_dir,  base_filename, clip_name, f"{clip_name}_clean.txt")
            out_suffix = 'overall'
        else:
            gt_file = os.path.join(gt_base_dir, base_filename, clip_name, f"{clip_name}_{suffix}_clean.txt")
            wx_file = os.path.join(wx_base_dir,  base_filename, clip_name, f"{clip_name}_{suffix}_clean.txt")
            out_suffix = suffix.lower()

        if not os.path.exists(gt_file) or not os.path.exists(wx_file):
            print(f"  Skipping {speaker_display} - file(s) not found")
            continue

        result = process_speaker(gt_file, wx_file, speaker_display, language)
        if result:
            all_results[speaker_display] = result
            report_file = os.path.join(output_dir, f"{clip_name}_{out_suffix}_evaluation.txt")
            html_file   = os.path.join(output_dir, f"{clip_name}_{out_suffix}_evaluation.html")
            write_detailed_report(result, report_file)
            generate_html_report(result, html_file)

    if all_results:
        combined_file = os.path.join(output_dir, f"{clip_name}_combined_evaluation.txt")
        write_combined_report(all_results, combined_file)

    print(f"\nEvaluation complete for {clip_name}!")
    return True

def main():
    parser = argparse.ArgumentParser(description='Multi-Speaker Transcription Evaluation')
    parser.add_argument('--ground-truth-dir', required=True)
    parser.add_argument('--whisperx-dir',     required=True)
    parser.add_argument('--output-dir',       required=True)
    parser.add_argument('--language', default='English', choices=['English', 'Spanish'])
    args = parser.parse_args()

    print("Starting Transcription Evaluation...")
    print("=" * 80)

    if not os.path.exists(args.ground_truth_dir):
        print(f"Error: Ground truth directory not found: {args.ground_truth_dir}")
        return
    if not os.path.exists(args.whisperx_dir):
        print(f"Error: WhisperX directory not found: {args.whisperx_dir}")
        return

    # Build matching recording/clip pairs
    gt_clips = set()
    for rec in os.listdir(args.ground_truth_dir):
        rec_path = os.path.join(args.ground_truth_dir, rec)
        if os.path.isdir(rec_path):
            for clip in os.listdir(rec_path):
                if os.path.isdir(os.path.join(rec_path, clip)):
                    gt_clips.add((rec, clip))

    wx_clips = set()
    for rec in os.listdir(args.whisperx_dir):
        rec_path = os.path.join(args.whisperx_dir, rec)
        if os.path.isdir(rec_path):
            for clip in os.listdir(rec_path):
                if os.path.isdir(os.path.join(rec_path, clip)):
                    wx_clips.add((rec, clip))

    matching = sorted(gt_clips & wx_clips)

    if not matching:
        print("No matching recording/clip pairs found!")
        print(f"  GT  has {len(gt_clips)} clips")
        print(f"  WX  has {len(wx_clips)} clips")
        return

    print(f"Found {len(matching)} matching clip(s):")
    for rec, clip in matching:
        print(f"  {rec} / {clip}")

    success_count = 0
    failed = []

    for base_filename, clip_name in matching:
        try:
            success = process_single_file(
                base_filename, clip_name,
                args.ground_truth_dir, args.whisperx_dir,
                args.output_dir, args.language
            )
            if success:
                success_count += 1
        except Exception as e:
            print(f"\nError processing {base_filename}/{clip_name}: {e}")
            import traceback; traceback.print_exc()
            failed.append(f"{base_filename}/{clip_name}")

    print("\n" + "=" * 80)
    print("ALL EVALUATIONS COMPLETE!")
    print(f"  Total:    {len(matching)}")
    print(f"  Success:  {success_count}")
    print(f"  Failed:   {len(failed)}")
    if failed:
        for f in failed:
            print(f"    - {f}")
    print(f"  Reports saved in: {args.output_dir}/")
    print("=" * 80)

if __name__ == "__main__":
    main()
