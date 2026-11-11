"""Per-query pipeline timeline from an I/O trace with channel samples.

usage: pipeline.py <trace.tsv> <query-index> [<query-index> ...]

Per channel: messages, first/last message time, and the share of its active time a
message sat in it waiting (its consumer was behind). With the last byte and the end
of the query, this shows which operator the tail waits on.
"""

import collections
import sys

trace = sys.argv[1]
queries = [int(q) for q in sys.argv[2:]]
labels = {}
for line in open(trace + ".chans"):
    cid, q, label = line.rstrip("\n").split("\t")
    labels[int(cid)] = (int(q), label)

chan = collections.defaultdict(list)  # cid -> [(t, messages, occupancy KB)]
landed = collections.defaultdict(float)
inflight = collections.defaultdict(list)
for line in open(trace):
    p = line.rstrip("\n").split("\t")
    if p[0] == "query":
        continue
    q, t, ev = int(p[0]), float(p[1]), p[2]
    if ev == "chan":
        chan[int(p[3])].append((t, int(p[4]), int(p[5])))
    elif ev == "landed":
        landed[q] = max(landed[q], t)
    elif ev == "iostat":
        inflight[q].append((t, int(p[5]) >> 32))

for q in queries:
    rows = []
    end = 0.0
    for cid, samples in chan.items():
        if labels.get(cid, (None,))[0] != q or not samples:
            continue
        samples.sort()
        first_msg = next((t for t, m, _ in samples if m > 0), None)
        last_t = samples[-1][0]
        end = max(end, last_t)
        msgs = samples[-1][1]
        busy = 0.0
        for (t0, _, occ), (t1, _, _) in zip(samples, samples[1:]):
            if occ > 0:
                busy += t1 - t0
        span = (last_t - first_msg) if first_msg is not None else 0.0
        rows.append((labels[cid][1], msgs, first_msg or 0.0, last_t, busy / span if span > 0 else 0.0))
    io = inflight[q]
    fetch_busy = sum(1 for _, mb in io if mb > 0) * 0.01
    print(f"== query index {q}: last byte {landed[q]:.2f}s, channels quiet at {end:.2f}s, "
          f"network in flight {fetch_busy:.1f}s of samples")
    print(f"   {'channel':42s} {'msgs':>5s} {'first':>6s} {'last':>6s} {'waiting':>8s}")
    for label, msgs, first, last, busy in sorted(rows, key=lambda r: r[3]):
        if msgs == 0:
            continue
        print(f"   {label:42s} {msgs:5d} {first:6.2f} {last:6.2f} {busy:7.0%}")
