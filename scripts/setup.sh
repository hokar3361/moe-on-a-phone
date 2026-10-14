#!/usr/bin/env bash
# Fetch llama.cpp at the tested commit, apply this project's patch and build the iOS xcframework.
# usage: scripts/setup.sh            (uses third_party/llama.cpp)
#        LLAMA_DIR=/path scripts/setup.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LLAMA_DIR="${LLAMA_DIR:-$ROOT/third_party/llama.cpp}"
COMMIT=fcac19079f3f2985fe74ed845f0880e7e9a8030a
PATCH="$ROOT/patches/llama.cpp/0001-moe-cache-direct-pread.patch"

if [[ ! -e "$LLAMA_DIR/.git" ]]; then
    git clone --filter=blob:none https://github.com/ggml-org/llama.cpp "$LLAMA_DIR"
fi
cd "$LLAMA_DIR"
if git apply --reverse --check "$PATCH" 2>/dev/null; then
    echo "patch already applied"
else
    git fetch origin "$COMMIT" 2>/dev/null || true
    git checkout -q "$COMMIT"
    git apply "$PATCH"
fi
LLAMA_BUILD_COMMON=ON ./build-xcframework.sh ios-device ios-sim
echo "built $LLAMA_DIR/build-apple/llama.xcframework"
