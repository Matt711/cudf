# usage: svc_cpu.sh <label> [env...]: runs svc_rate.py (REPS=3) and reports its CPU over the whole run
S=/tmp/claude-1000/-home-ubuntu-mmurray2-worktrees-tpch-sf1000-baseline/be48e216-ca82-4509-95d9-e8197ac578d5/scratchpad
label=$1; shift
env "$@" REPS=3 /home/ubuntu/mmurray2/envs/io2/bin/python $S/svc_rate.py $label > $S/svc_$label.out 2>&1 & PID=$!
until grep -q rep0 $S/svc_$label.out 2>/dev/null || ! kill -0 $PID 2>/dev/null; do sleep 0.1; done
read u0 s0 < <(awk '{print $14, $15}' /proc/$PID/stat); t0=$(date +%s.%N)
until grep -q rep2 $S/svc_$label.out 2>/dev/null || ! kill -0 $PID 2>/dev/null; do sleep 0.05; done
read u1 s1 < <(awk '{print $14, $15}' /proc/$PID/stat 2>/dev/null); t1=$(date +%s.%N)
wait $PID; grep "rep[12]" $S/svc_$label.out | tr '\n' ' '
python3 -c "print('| CPU user %.1f sys %.1f cores' % (($u1-$u0)/100/($t1-$t0), ($s1-$s0)/100/($t1-$t0)))"
