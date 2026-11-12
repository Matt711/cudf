"""Machine-wide CPU split once a second (cores busy as user, system, softirq, irq, iowait) until <seconds>. usage: stat_sample.py <out.tsv> <seconds>"""
import sys, time
out, dur = sys.argv[1], float(sys.argv[2])
def read():
    f = open("/proc/stat").readline().split()[1:]
    return [int(x) for x in f]
names = ["user", "nice", "system", "idle", "iowait", "irq", "softirq", "steal"]
prev = read(); t0 = time.time()
with open(out, "w") as o:
    o.write("t_s\t" + "\t".join(names) + "\n")
    while time.time() - t0 < dur:
        time.sleep(1.0); cur = read(); d = [c - p for c, p in zip(cur, prev)]; tot = sum(d[:8]) or 1
        o.write(f"{time.time()-t0:.1f}\t" + "\t".join(f"{32*d[i]/tot:.2f}" for i in range(8)) + "\n"); o.flush(); prev = cur
