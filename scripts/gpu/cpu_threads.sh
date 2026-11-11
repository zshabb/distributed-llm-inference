#!/usr/bin/env bash
# Print the number of CPUs this process may actually use: the cgroup CPU quota (v2 cpu.max or
# v1 cfs_quota_us / cfs_period_us), rounded down, else nproc. Containers often see every host
# thread (e.g. 96 on a RunPod node with a 7.65-CPU quota); libraries that size thread pools from
# the visible count then oversubscribe and get throttled. Runs on the GPU host; reads only.
set -euo pipefail
root="${CGROUP_ROOT:-/sys/fs/cgroup}"
quota="" period=""
if [[ -r "$root/cpu.max" ]]; then
  read -r quota period < "$root/cpu.max"
elif [[ -r "$root/cpu/cpu.cfs_quota_us" && -r "$root/cpu/cpu.cfs_period_us" ]]; then
  quota="$(cat "$root/cpu/cpu.cfs_quota_us")"
  period="$(cat "$root/cpu/cpu.cfs_period_us")"
fi
visible="$(nproc)"
if [[ -n "$quota" && "$quota" != "max" && "$quota" -gt 0 && "${period:-0}" -gt 0 ]]; then
  n=$(( quota / period ))
  (( n < 1 )) && n=1
  (( n > visible )) && n=$visible
  echo "$n"
else
  echo "$visible"
fi
