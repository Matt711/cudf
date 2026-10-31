"""1 Hz CPU by thread family for one process: thread_cpu_sampler.py <pid> <out.csv>. Columns: epoch, comm, nthreads, user_pct, sys_pct (percent of one core, summed over the family)."""
import os, sys, time, collections
pid, out = sys.argv[1], sys.argv[2]
tck = os.sysconf("SC_CLK_TCK")
def snap():
    d = {}
    try: tids = os.listdir(f"/proc/{pid}/task")
    except FileNotFoundError: return None
    for t in tids:
        try:
            s = open(f"/proc/{pid}/task/{t}/stat").read()
        except OSError: continue
        l, r = s.index("("), s.rindex(")")
        comm = s[l+1:r]; f = s[r+2:].split()
        d[t] = (comm, int(f[11]), int(f[12]))   # utime, stime (ticks)
    return d
prev = snap(); tprev = time.time()
with open(out, "w") as fo:
    fo.write("epoch,comm,nthreads,user_pct,sys_pct\n")
    while True:
        time.sleep(1.0)
        cur = snap(); now = time.time()
        if cur is None: break
        dt = now - tprev; agg = collections.defaultdict(lambda: [0, 0.0, 0.0])
        for t, (c, u, s) in cur.items():
            pu, ps = (prev.get(t, (c, u, s))[1:]) if t in prev else (u, s)
            a = agg[c]; a[0] += 1; a[1] += (u - pu) / tck / dt * 100; a[2] += (s - ps) / tck / dt * 100
        for c, (n, u, s) in agg.items():
            if u + s > 1: fo.write(f"{int(now)},{c},{n},{u:.0f},{s:.0f}\n")
        prev, tprev = cur, now
