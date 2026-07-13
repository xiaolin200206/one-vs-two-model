#!/usr/bin/env bash
# Pin the Raspberry Pi 5 CPU governor to `performance` before benchmarking.
#
# Why: with the default `ondemand`/`schedutil` governor, a frequency drop is
# ambiguous -- it could be power-saving downclocking OR thermal throttling.
# Pinning to `performance` removes the first possibility, so any frequency
# reduction observed during a run is thermal, which is what the paper claims.
#
#   sudo ./lock_governor.sh          # pin
#   sudo ./lock_governor.sh restore  # back to ondemand
#
# Run cache_benchmark.py WITHOUT sudo afterwards, so perf uses the :u
# (user-space) counters.

set -euo pipefail
MODE="${1:-performance}"
[ "$MODE" = "restore" ] && MODE="ondemand"

for c in /sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_governor; do
    echo "$MODE" > "$c"
done

echo "governor set to: $MODE"
echo -n "current: "; cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
echo -n "freq   : "; cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq
echo -n "throttle: "; vcgencmd get_throttled
