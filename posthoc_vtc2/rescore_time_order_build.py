#!/usr/bin/env python3
# Rebuild each clip's OVERALL hypothesis in time order from {clip}_tagged.txt (UNKNOWN/SPEECH
# already dropped). First checks that grouping the same lines by speaker gives back the existing
# _clean.txt exactly, then writes a view whose _clean.txt is in time order (other files symlinked).
#
#   python rescore_time_order_build.py SRC_EVAL_VIEW DST_VIEW
import os,re,sys,glob
if len(sys.argv) != 3: sys.exit('usage: rescore_time_order_build.py SRC_EVAL_VIEW DST_VIEW')
src,dst=sys.argv[1],sys.argv[2]
def clean(t):
    t=re.sub(r'\[.*?\]','',t); t=re.sub(r'\bxxx\b','',t,flags=re.I); return ' '.join(t.split()).strip()
ok=bad=0
for cd in sorted(glob.glob(src+'/*/*/')):
    clip=os.path.basename(cd.rstrip('/')); rec=os.path.basename(os.path.dirname(cd.rstrip('/')))
    od=f'{dst}/{rec}/{clip}'; os.makedirs(od,exist_ok=True)
    for f in os.listdir(cd):
        if f!=f'{clip}_clean.txt': os.symlink(os.path.realpath(cd+f),f'{od}/{f}')
    tf=cd+f'{clip}_tagged.txt'
    if not os.path.exists(tf): continue
    items=[]
    for l in open(tf,encoding='utf-8'):
        m=re.match(r'\[[^\]]+\]\s*\(\w+\)\s*\[(\w+)\]:\s*(.*)',l.strip())
        if m: items.append((m.group(1),clean(m.group(2)) or '[unknown]'))
    order=['KCHI','OCH','FEM','MAL']+[s for s in dict.fromkeys(s for s,_ in items) if s not in ('KCHI','OCH','FEM','MAL')]
    grouped=' '.join(t for s in order for sp,t in items if sp==s)
    pub=open(cd+f'{clip}_clean.txt',encoding='utf-8').read() if os.path.exists(cd+f'{clip}_clean.txt') else ''
    if grouped==pub: ok+=1
    else: bad+=1; print('MISMATCH',clip)
    open(f'{od}/{clip}_clean.txt','w',encoding='utf-8').write(' '.join(t for _,t in items))
print('grouped reconstruction matches published _clean.txt:',ok,'mismatch:',bad)
