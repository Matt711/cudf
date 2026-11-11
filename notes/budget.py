"""Per-query time budget from an I/O trace: startup before the first read, fetch span, tail after the last byte, and
arena-limited time (arena under 2 GB free while reads in flight are below 90% of the cap)."""
import collections, re, sys
trace, runlog, cap_mb = sys.argv[1], sys.argv[2], int(sys.argv[3])
times = [float(m[2]) for m in (re.search(r"Query (\d+) - Iteration (\d+) finished in ([\d.]+)s", l) for l in open(runlog)) if m for m in [(m[1], m[2], m[3])] if True for m in [m]]
times = []
for l in open(runlog):
    m = re.search(r"Query (\d+) - Iteration (\d+) finished in ([\d.]+)s", l)
    if m: times.append((int(m[1]), int(m[2]), float(m[3])))
io = collections.defaultdict(list); landed = collections.defaultdict(float); fetched = {}
for line in open(trace):
    p = line.rstrip("\n").split("\t")
    if p[0] == "query": continue
    q, t, ev = int(p[0]), float(p[1]), p[2]
    if ev == "iostat":
        v = int(p[5]); io[q].append((t, v >> 32, v & 0xFFFFFFFF))
    elif ev == "landed": landed[q] = max(landed[q], t)
    elif ev == "fetched_total": fetched[q] = int(p[5])
print(" q it   total  start  fetch   tail  arena-limited  GB")
S = [0.0]*5
for i, (qn, it, total) in enumerate(times):
    s = sorted(io[i]); 
    first = next((t for t, inf, _ in s if inf > 0), 0.0)
    lim = sum(0.01 for t, inf, free in s if free < 2048 and inf < 0.9 * cap_mb and t < landed[i])
    gb = (fetched.get(i + 1, 0) - fetched.get(i, 0)) / 1e9 if i + 1 in fetched else float("nan")
    tail = total - landed[i]
    row = [total, first, landed[i] - first, tail, lim]
    S = [a + b for a, b in zip(S, row)]
    if it == 0:
        print(f"{qn:2d} {it}  {total:6.2f} {first:6.2f} {landed[i]-first:6.2f} {tail:6.2f}  {lim:6.2f}  {gb:7.1f}")
