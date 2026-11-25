#!/usr/bin/env bash
# Launch every (arm x share) config in parallel from YOUR terminal (not from an agent session,
# which would kill the jobs when it ends). Each config is resumable: re-run to continue.
#
#   cd "v2_event_study" && source .venv/bin/activate && ./run_all.sh
#   PARALLEL=4 ./run_all.sh        # limit concurrent configs (default 10 = all)
# Works with macOS's old bash 3.2 (no `wait -n`).
set -u
cd "$(dirname "$0")"
PARALLEL="${PARALLEL:-10}"
mkdir -p out/logs

run_one() {
  local arm=$1 share=$2
  local log="out/logs/${arm}_share${share}.log"
  echo "starting $arm share=$share -> $log"
  python -m herd2.run run --arm "$arm" --share "$share" > "$log" 2>&1
  echo "finished $arm share=$share (exit $?)"
}

throttle() {
  while [ "$(jobs -rp | wc -l | tr -d ' ')" -ge "$PARALLEL" ]; do sleep 5; done
}

run_one mono 0.0 &            # shared share-0 control, run once
for arm in mono persona mixed; do
  for share in 0.25 0.5 1.0; do
    throttle
    run_one "$arm" "$share" &
  done
done
wait
echo "all configs finished. Check out/logs/*.log for ABORT lines, then: python -m herd2.analyze"
