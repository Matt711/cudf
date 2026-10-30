# Prototype tool: per-column byte prices from parquet footers + simulation of cache policies (see DESIGN.md section 2).
# Expects results/needed-bytes.json (per-query projected columns) and the local TPC-H SF1000 parquet copy at /datasets/mmurray2/tpch-sf1000.
"""Column-level cache simulation from parquet footers + per-query projected columns (results/needed-bytes.json).
Queries run in order q1..q22. Access = (table, column) set per query. Sizes = compressed column bytes over all files."""
import glob, json, sys
from collections import defaultdict, OrderedDict
import pyarrow.parquet as pq
ROOT="/datasets/mmurray2/tpch-sf1000"
n=json.load(open('results/needed-bytes.json'))
sizes={}
for t in {p['table'] for r in n.values() for p in r['scans']}:
    d=defaultdict(int)
    for f in glob.glob(f"{ROOT}/{t}/*.parquet"):
        m=pq.ParquetFile(f).metadata
        for g in range(m.num_row_groups):
            rg=m.row_group(g)
            for c in range(rg.num_columns): d[rg.column(c).path_in_schema]+=rg.column(c).total_compressed_size
    for c,b in d.items(): sizes[(t,c)]=b
acc=[]  # per query: set of (table,col)
for q in range(1,23):
    s=set()
    for p in n[str(q)]['scans']:
        for c in p['cols']: s.add((p['table'],c))
    acc.append(s)
GB=1e9
uses=defaultdict(list)
for qi,s in enumerate(acc):
    for k in s: uses[k].append(qi+1)
nocache=sum(sizes[k] for s in acc for k in s)
distinct=sum(sizes[k] for k in uses)
print(f"no cache {nocache/GB:.0f} GB; infinite cache (read each distinct column once) {distinct/GB:.0f} GB; {len(uses)} distinct columns\n")
print("Top columns by bytes re-read (size x (uses-1)):")
print("| table.column | size GB | used by | uses | GB read without cache | GB saved if cached |\n|---|---|---|---|---|---|")
rows=sorted(uses.items(), key=lambda kv: -sizes[kv[0]]*(len(kv[1])-1))
for k,u in rows[:16]:
    print(f"| {k[0]}.{k[1]} | {sizes[k]/GB:.1f} | {','.join('q%d'%x for x in u)} | {len(u)} | {sizes[k]*len(u)/GB:.1f} | {sizes[k]*(len(u)-1)/GB:.1f} |")
def lru(C):
    cache=OrderedDict(); used=0; read=0
    for s in acc:
        for k in sorted(s, key=lambda k:-sizes[k]):  # order within query arbitrary; touch all
            sz=sizes[k]
            if k in cache: cache.move_to_end(k); continue
            read+=sz
            if sz>C: continue
            while used+sz>C:
                ek,_=cache.popitem(last=False); used-=sizes[ek]
            cache[k]=sz; used+=sz
    return read
def belady(C):
    # evict the cached column whose next use is furthest (or never)
    cache={}; used=0; read=0
    for qi,s in enumerate(acc):
        need=set(s)
        for k in s:
            sz=sizes[k]
            if k in cache: continue
            read+=sz
            if sz>C: continue
            while used+sz>C:
                def nxt(x):
                    for j in range(qi+1,22):
                        if x in acc[j]: return j
                    return 99
                cand=[x for x in cache if x not in need]
                if not cand: break
                ev=max(cand,key=nxt); used-=cache.pop(ev)
            if used+sz<=C: cache[k]=sz; used+=sz
    return read
def static(C):
    chosen=[]; used=0
    for k,u in sorted(uses.items(), key=lambda kv:(-(len(kv[1])-1), sizes[kv[0]])):
        if len(u)>1 and used+sizes[k]<=C: chosen.append(k); used+=sizes[k]
    cs=set(chosen)
    read=sum(sizes[k] for s in acc for k in s if k not in cs)+sum(sizes[k] for k in cs)
    return read,used
print("\nBytes read from S3 by cache policy (column granularity, queries in order q1..q22):")
print("| cache size GB | LRU | best static pin set | Belady (oracle replacement) |\n|---|---|---|---|")
for C in (10,25,50,75,100,125,150,186):
    c=C*GB; st,_=static(c)
    print(f"| {C} | {lru(c)/GB:.0f} | {st/GB:.0f} | {belady(c)/GB:.0f} |")
# which columns would a 75 GB static pin choose
C=75*GB; ch=[]; used=0
for k,u in sorted(uses.items(), key=lambda kv:(-(len(kv[1])-1), sizes[kv[0]])):
    if len(u)>1 and used+sizes[k]<=C: ch.append(k); used+=sizes[k]
print("\n75 GB static pin set:", ", ".join(f"{k[0]}.{k[1]}({sizes[k]/GB:.1f})" for k in ch))
json.dump({f"{k[0]}.{k[1]}":{"bytes":sizes[k],"queries":u} for k,u in uses.items()}, open('results/column-usage.json','w'), indent=1)

def online(C, kind):
    """Online policies at column granularity. kind: 'lfu' (evict fewest hits, tie oldest), 'gdsf' (evict lowest freq/size), 'lru2' (evict oldest second-to-last use)."""
    cache={}  # k -> [freq, last, prev_last]
    used=0; read=0; tick=0
    for s in acc:
        for k in sorted(s, key=lambda k:-sizes[k]):
            tick+=1; sz=sizes[k]
            if k in cache:
                e=cache[k]; e[0]+=1; e[2]=e[1]; e[1]=tick; continue
            read+=sz
            if sz>C: continue
            while used+sz>C:
                cand=[x for x in cache if x not in s]
                if not cand: break
                if kind=='lfu': ev=min(cand,key=lambda x:(cache[x][0],cache[x][1]))
                elif kind=='gdsf': ev=min(cand,key=lambda x:(cache[x][0]/sizes[x],cache[x][1]))
                else: ev=min(cand,key=lambda x:(cache[x][2],cache[x][1]))
                used-=sizes[ev]; del cache[ev]
            if used+sz<=C: cache[k]=[1,tick,0]; used+=sz
    return read
print("\nOnline policies, bytes read (GB):")
print("| cache GB | LRU | LRU-2 | LFU | size-aware LFU (GDSF) | static oracle |\n|---|---|---|---|---|---|")
for C in (25,50,75,100,125,150):
    c=C*GB
    print(f"| {C} | {lru(c)/GB:.0f} | {online(c,'lru2')/GB:.0f} | {online(c,'lfu')/GB:.0f} | {online(c,'gdsf')/GB:.0f} | {static(c)[0]/GB:.0f} |")
