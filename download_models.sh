#!/usr/bin/env bash
# Download the pinned GGUF files used in this project and verify sha256.
# Resumable: re-run after an interruption.
set -euo pipefail
M="$HOME/models"

fetch() { # repo rev path dest sha256
    local url="https://huggingface.co/$1/resolve/$2/$3"
    local dest="$4" sha="$5"
    if [[ -f "$dest.ok" ]]; then echo "ok (cached): $dest"; return; fi
    mkdir -p "$(dirname "$dest")"
    curl -L --fail --retry 20 --retry-delay 5 -C - -o "$dest" "$url"
    echo "verifying $dest"
    echo "$sha  $dest" | shasum -a 256 -c - && touch "$dest.ok"
}

Q=unsloth/Qwen3.6-35B-A3B-MTP-GGUF
QR=5bc3e238d916f48a861bac2f8a1990a0e9b7e98d
fetch $Q $QR Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf "$M/qwen36-35b/Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf" 627e2b05f83088448e27861ec392d56efc2b178783cae5adfa4ad2d988441203
fetch $Q $QR Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf  "$M/qwen36-35b/Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf"  ed7cda7e38985b4fcff76475865135039641d2bfbac3c169df15ca770f37fb0c

F=unsloth/Qwen3.8-Flash-Next-GGUF
FR=766911a6b7369840a91dbcd95f9f997acaab6cd6
fetch $F $FR UD-IQ1_S/Qwen3.8-Flash-Next-UD-IQ1_S-00001-of-00003.gguf "$M/qwen38-flash-next/Qwen3.8-Flash-Next-UD-IQ1_S-00001-of-00003.gguf" 88a1420825a9304063e882ada29d438263617f51ac8923d438d927496693bafd
fetch $F $FR UD-IQ1_S/Qwen3.8-Flash-Next-UD-IQ1_S-00002-of-00003.gguf "$M/qwen38-flash-next/Qwen3.8-Flash-Next-UD-IQ1_S-00002-of-00003.gguf" 3a62e35bbf9add4733bd1438ebd3a67649d5edd6cb0e72bb78e33c913992b2b6
fetch $F $FR UD-IQ1_S/Qwen3.8-Flash-Next-UD-IQ1_S-00003-of-00003.gguf "$M/qwen38-flash-next/Qwen3.8-Flash-Next-UD-IQ1_S-00003-of-00003.gguf" 0e25ceaeb89b8a80aa973c6c0c7448943682f7408c2855b2ebd016b7643a861a
echo ALL DONE
