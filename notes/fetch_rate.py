"""Per-second fetch view of one query: landed GB/s per stream, mean in-flight/queued/free MB."""
import collections, sys
trace, q = sys.argv[1], int(sys.argv[2])
land = collections.defaultdict(lambda: collections.defaultdict(int))
st = collections.defaultdict(list)
streams = {}
for line in open(trace):
    p = line.rstrip("\n").split("\t")
    if p[0] == "query" or int(p[0]) != q:
        continue
    t, ev = float(p[1]), p[2]
    b = int(t)
    if ev == "landed":
        land[b][int(p[3])] += int(p[5])
    elif ev == "scan_start":
        streams[int(p[3])] = int(p[5])
    elif ev == "iostat":
        v = int(p[5])
        st[b].append((v >> 32, int(p[4]), v & 0xFFFFFFFF))
ids = sorted(streams)
print("sec " + " ".join(f"s{i}({streams[i]}t)".rjust(10) for i in ids) + "   total  inflMB queuMB freeMB")
for b in sorted(set(land) | set(st)):
    row = [land[b].get(i, 0) / 1e9 for i in ids]
    s = st[b] or [(0, 0, 0)]
    m = [sum(x[k] for x in s) / len(s) for k in range(3)]
    print(f"{b:3d} " + " ".join(f"{r:10.2f}" for r in row) + f"  {sum(row):6.2f}  {m[0]:6.0f} {m[1]:6.0f} {m[2]:6.0f}")
