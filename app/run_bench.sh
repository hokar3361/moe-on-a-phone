#!/usr/bin/env bash
# Run the app on the connected iPhone with the given args; print result lines.
# usage: DEV=<udid> BUNDLE=<bundle id> OUT=<dir> app/run_bench.sh <log-name> <app args...>
#   DEV: from `xcrun devicectl list devices`; BUNDLE: the PRODUCT_BUNDLE_IDENTIFIER in Config/Local.xcconfig
set -u
DEV=${DEV:?set DEV to the device identifier}
BUNDLE=${BUNDLE:-com.example.moebench}
OUT=${OUT:-/tmp}
log="$OUT/$1.log"; shift
xcrun devicectl device process launch --console --terminate-existing --device "$DEV" "$BUNDLE" "$@" > "$log" 2>&1 &
pid=$!
until grep -qE "^DONE|^ERROR" "$log" 2>/dev/null || ! kill -0 $pid 2>/dev/null; do sleep 2; done
sleep 2; kill $pid 2>/dev/null
grep -E "^(\[|ERROR|thermal)|^mem: |block|llama_moe_cache: (time|ubatch <=|predict)|^spec:|^spin|not mapped|MoE cache size|MoE bell on|MTL0_Mapped|mmap failed|error loading|metal-timing|llama_moe_prefetch: mode" "$log"
