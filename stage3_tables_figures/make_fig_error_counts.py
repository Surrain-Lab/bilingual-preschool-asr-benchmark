#!/usr/bin/env python3
"""Error counts for VTC2 + WhisperX by speaker and by language
(fig_error_counts_by_speaker_language.png).

Blue = substitutions, hatched orange = deletions, cross-hatched white =
insertions. Counts are totals over the 14 recordings (157 scored clips):
  speaker panel  -> the per-clip evaluation reports from stage 1
  language panel -> stage2_aggregate/wer_by_language.py on vtc2/langdetect
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv, os

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'figures_out')

# ---- counts: (label, substitutions, deletions, insertions, reference words) ----
TALKER = [
    ('Key\nchild',      4054, 4930,  489, 13218),
    ('Male\nadult',     1720, 2677,  231,  7040),
    ('Female\nadult',   3337, 5359,  663, 12675),
    ('Adult\ncombined', 5329, 7665,  649, 19715),
]
LANG = [
    ('Spanish', 7469, 12128,  854, 30023),
    ('English', 1617,   530, 2885,  3622),
]

SUB, DEL, INS = '#0072B2', '#E69F00', '#F2F2F2'   # Okabe-Ito blue / orange / near-white
HATCH = ['', '///', 'xxx']

plt.rcParams.update({'font.size': 13, 'axes.linewidth': 1.0,
                     'hatch.linewidth': 0.8, 'figure.dpi': 200})

fig, (ax1, ax2) = plt.subplots(
    1, 2, figsize=(15.4, 7.1), gridspec_kw={'width_ratios': [2.05, 1]})

YMAX = 22000

def draw(ax, data, title):
    xs = range(len(data))
    bottoms = [0] * len(data)
    parts = [('Substitutions', SUB, 1), ('Deletions', DEL, 2), ('Insertions', INS, 3)]
    for (name, color, idx), hatch in zip(parts, HATCH):
        vals = [d[idx] for d in data]
        ax.bar(xs, vals, bottom=bottoms, width=0.55, color=color, hatch=hatch,
               edgecolor='black', linewidth=1.0, label=name, zorder=3)
        bottoms = [b + v for b, v in zip(bottoms, vals)]
    for x, (d, tot) in enumerate(zip(data, bottoms)):
        ax.text(x, tot + YMAX * 0.016, f'{tot:,}', ha='center', va='bottom',
                fontweight='bold', fontsize=14, zorder=4)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([f'{d[0]}\n({d[4]:,} ref. words)' for d in data],
                       fontsize=12.5)
    ax.set_ylim(0, YMAX)
    ax.set_yticks(range(0, YMAX + 1, 2000))
    ax.set_yticklabels([f'{v:,}' for v in range(0, YMAX + 1, 2000)])
    ax.set_title(title, fontsize=15, pad=14)
    ax.grid(axis='y', color='#DDDDDD', linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis='both', length=4)

draw(ax1, TALKER, 'By speaker')
draw(ax2, LANG, 'By assigned language')
ax1.set_ylabel('Number of error words', fontsize=14)
ax2.tick_params(labelleft=True)

h, l = ax1.get_legend_handles_labels()
fig.legend(h, l, loc='upper center', ncol=3, frameon=False,
           fontsize=14, bbox_to_anchor=(0.5, 1.0), columnspacing=2.6)
fig.tight_layout(rect=[0, 0, 1, 0.92])

for ext in ('png', 'pdf'):
    p = os.path.join(OUT, f'fig_error_counts_by_speaker_language.{ext}')
    fig.savefig(p, bbox_inches='tight', facecolor='white')
    print('wrote', p)

# ---- the counts, as a table, and the rate check ----
with open(os.path.join(OUT, 'error_counts_by_speaker_language.csv'), 'w',
          newline='') as fh:
    w = csv.writer(fh)
    w.writerow(['panel', 'group', 'ref_words', 'substitutions', 'deletions',
                'insertions', 'total_errors', 'pooled_wer'])
    for panel, data in (('talker', TALKER), ('language', LANG)):
        for lab, s, d, i, rw in data:
            w.writerow([panel, lab.replace('\n', ' '), rw, s, d, i, s + d + i,
                        f'{(s + d + i) / rw:.4f}'])
print('\n{:<26}{:>9}{:>9}{:>9}{:>8}{:>10}{:>9}'.format(
    'group', 'ref', 'subs', 'dels', 'ins', 'total', 'pooledWER'))
print('-' * 80)
for panel, data in (('talker', TALKER), ('language', LANG)):
    for lab, s, d, i, rw in data:
        t = s + d + i
        print(f'{lab.replace(chr(10), " "):<26}{rw:>9,}{s:>9,}{d:>9,}{i:>8,}'
              f'{t:>10,}{t / rw:>9.3f}')
