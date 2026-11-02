"""Run a command and sample per-thread CPU (from /proc) every 0.5 s while it runs; print windows where some thread group is busy."""
import subprocess, sys, time, os, collections
p = subprocess.Popen(sys.argv[1:], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
hz = os.sysconf("SC_CLK_TCK")
def snap():
    d = {}
    try:
        for t in os.listdir(f"/proc/{p.pid}/task"):
            try:
                st = open(f"/proc/{p.pid}/task/{t}/stat").read()
                comm = st[st.index("(") + 1: st.rindex(")")]; f = st[st.rindex(")") + 2:].split()
                d[t] = (comm, int(f[11]) + int(f[12]))
            except Exception: pass
    except Exception: pass
    return d
prev = snap(); tp = time.time()
while p.poll() is None:
    time.sleep(0.5); cur = snap(); now = time.time(); dt = now - tp
    by = collections.defaultdict(lambda: [0.0, 0, 0.0])
    for t, (c, v) in cur.items():
        if t in prev:
            u = (v - prev[t][1]) / hz / dt * 100; by[c][0] += u; by[c][1] += 1; by[c][2] = max(by[c][2], u)
    top = sorted(by.items(), key=lambda x: -x[1][0])[:4]
    if top and top[0][1][0] > 120:
        print(f"t={now - tp:5.1f}s ", " | ".join(f"{c}[{n}] sum={u:.0f}% max1={m:.0f}%" for c, (u, n, m) in top))
    prev, tp = cur, now
out = p.communicate()[0]; print([l for l in out.splitlines() if l.startswith("threads=")][-1])
