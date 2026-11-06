#!/usr/bin/env bash
set -euo pipefail

mkdir -p .data/goal/raw_sources

CANDIDATES=(
    "k6ObNLE9fpw"
    "upaVQXpSMBw"
    "J54Nbyy-ExI"
    "va-Dop0YDZI"
    "Aj3ATU0BVZo"
    "aRKO1B_piTo"
    "F6Rr_6eRGSA"
    "XlTMPg8zb8g"
    "_UjcxyKrsvE"
    "p_u50y858kY"
    "wFE3JfBNYhc"
    "wJsmmiXd-nw"
)

for vid in "${CANDIDATES[@]}"; do
    out=".data/goal/raw_sources/${vid}.wav"
    if [[ -f "$out" ]]; then
        echo "Already downloaded: $out"
        continue
    fi
    echo "=== Downloading $vid ==="
    scripts/s1-download/youtube.sh -u "$vid" -sr 16000 -o "$out" --sleep-requests 1.0 --sleep-interval 1.0 --max-sleep-interval 3.0
    echo "Done $vid -> $out"
done

echo "All candidates downloaded successfully."
