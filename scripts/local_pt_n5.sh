#!/usr/bin/env bash
# local_pt_n5.sh — Local PT tuning workflow for N=5
#
# Usage:
#   ./scripts/local_pt_n5.sh [mode] [options]
#
# Modes:
#   debug      One PT run with PT_DEBUG output (default)
#   nodebug    One PT run without debug (production binary)
#   sweep      Compact grid: Tmax × K_epoch, fixed R=8, two seeds
#
# Sweep knobs (override via env):
#   TMAX_LIST     space-separated Tmax values    (default: "25 50 100")
#   K_LIST        space-separated pt_K_epoch     (default: "200 1000 5000")
#   TMIN          cold temperature                (default: 1.0)
#   R             replicas                        (default: 8)
#   BUDGET        seconds per run                 (default: 60)
#   SEEDS         space-separated seeds           (default: "42 7")
#   DEBUG         1=compile with PT_DEBUG 0=no   (default: 1 for debug mode)
#
# Examples:
#   ./scripts/local_pt_n5.sh                          # one debug run, defaults
#   ./scripts/local_pt_n5.sh debug                    # explicit debug run
#   ./scripts/local_pt_n5.sh nodebug                  # no-debug single run
#   ./scripts/local_pt_n5.sh sweep                    # full grid
#   TMAX_LIST="50 100" K_LIST="1000 5000" ./scripts/local_pt_n5.sh sweep
#   BUDGET=180 SEEDS="42" ./scripts/local_pt_n5.sh sweep

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

MODE="${1:-debug}"

# --- Tuning knobs with env overrides ---
TMIN="${TMIN:-1.0}"
R="${R:-8}"
BUDGET="${BUDGET:-60}"
SEEDS="${SEEDS:-42 7}"
TMAX_LIST="${TMAX_LIST:-25 50 100}"
K_LIST="${K_LIST:-200 1000 5000}"

BASE_CFLAGS="-O3 -std=c11 -Wall -Wextra -Iinclude -fopenmp"

# ---------------------------------------------------------------------------
build_debug() {
    echo "==> Building with PT_DEBUG..."
    make -B CFLAGS="$BASE_CFLAGS -DPT_DEBUG" 2>&1
}

build_nodebug() {
    echo "==> Building without PT_DEBUG..."
    make -B CFLAGS="$BASE_CFLAGS" 2>&1
}

run_one() {
    local tmin="$1" tmax="$2" k="$3" seed="$4" budget="$5"
    echo ""
    echo "--- PT N=5 R=$R Tmin=$tmin Tmax=$tmax K=$k seed=$seed budget=${budget}s ---"
    ./bin/solver --study --method pt --N 5 --R "$R" \
        --time_budget_sec "$budget" \
        --seed "$seed" --run_id 0 \
        --pt_Tmin "$tmin" --pt_Tmax "$tmax" --pt_K_epoch "$k" \
        2>&1
}

# ---------------------------------------------------------------------------
if [ "$MODE" = "debug" ]; then
    build_debug
    run_one "$TMIN" 25 1000 42 "$BUDGET"

elif [ "$MODE" = "nodebug" ]; then
    build_nodebug
    run_one "$TMIN" 25 1000 42 "$BUDGET"

elif [ "$MODE" = "sweep" ]; then
    build_nodebug

    # Header
    printf "\n%-8s %-8s %-8s %-6s %-8s %-8s %-8s %-8s %-8s %-6s\n" \
        "Tmax" "K_epoch" "seed" "probes" "feasible" "L_best" \
        "swp_att" "swp_acc" "rate" "time_s"
    printf '%s\n' "$(printf '%0.s-' {1..80})"

    for tmax in $TMAX_LIST; do
        for k in $K_LIST; do
            for seed in $SEEDS; do
                out=$(run_one "$TMIN" "$tmax" "$k" "$seed" "$BUDGET" 2>&1)

                # Parse from "Bisection complete: ..." line
                probes=$(echo "$out"  | grep -oP 'probes=\K[0-9]+' | tail -1 || echo "?")
                feasible=$(echo "$out" | grep -oP 'feasible=\K[0-9]+' | tail -1 || echo "?")
                Lbest=$(echo "$out"   | grep -oP 'L_best=\K[-0-9.]+' | tail -1 || echo "?")
                # Parse from "PT probe N: swap_attempts=... swap_accepts=..." (non-debug)
                att=$(echo "$out"  | grep -oP 'swap_attempts=\K[0-9]+' | paste -sd+ | bc 2>/dev/null || echo "?")
                acc=$(echo "$out"  | grep -oP 'swap_accepts=\K[0-9]+'  | paste -sd+ | bc 2>/dev/null || echo "?")
                if [ "$att" != "?" ] && [ "$att" != "0" ] && [ "$acc" != "?" ]; then
                    rate=$(awk "BEGIN{printf \"%.3f\", $acc/$att}")
                else
                    rate="?"
                fi

                printf "%-8s %-8s %-6s %-6s %-8s %-8s %-8s %-8s %-8s\n" \
                    "$tmax" "$k" "$seed" "$probes" "$feasible" "$Lbest" \
                    "$att" "$acc" "$rate"
            done
        done
    done

    echo ""
    echo "Sweep complete. Re-run with sweep_debug mode for distinctE and coldE stats."

# ---------------------------------------------------------------------------
# sweep_debug: build with PT_DEBUG, run Tmax x K_epoch grid, parse rich stats
# from "PT DEBUG SUMMARY" lines. Each config shows per-probe averages.
# ---------------------------------------------------------------------------
elif [ "$MODE" = "sweep_debug" ]; then
    echo "==> Building with PT_DEBUG..."
    make -B CFLAGS="$BASE_CFLAGS -DPT_DEBUG" 2>&1 | grep -v "^Compiling"

    printf "\n%-6s %-6s %-4s | %-5s %-5s %-6s | %-8s %-8s %-8s %-5s\n" \
        "Tmax" "K" "seed" "prob" "feas" "Lbest" "rate" "distinctE" "coldE" "feas_r"
    printf '%s\n' "$(printf '%0.s-' {1..80})"

    for tmax in $TMAX_LIST; do
        for k in $K_LIST; do
            for seed in $SEEDS; do
                out=$(run_one "$TMIN" "$tmax" "$k" "$seed" "$BUDGET" 2>&1)

                probes=$(echo "$out"   | grep -oP 'probes=\K[0-9]+'   | tail -1 || echo "?")
                feasible=$(echo "$out" | grep -oP 'feasible=\K[0-9]+' | tail -1 || echo "?")
                Lbest=$(echo "$out"    | grep -oP 'L_best=\K[-0-9.]+' | tail -1 || echo "?")

                # Average debug stats across all PT DEBUG SUMMARY lines
                debug_lines=$(echo "$out" | grep "PT DEBUG SUMMARY" || true)
                if [ -n "$debug_lines" ]; then
                    rate=$(echo "$debug_lines" | grep -oP 'rate=\K[0-9.]+' | awk '{s+=$1;n++} END{if(n>0) printf "%.3f",s/n; else print "?"}')
                    distE=$(echo "$debug_lines" | grep -oP 'distinctE=\K[0-9]+'  | awk '{s+=$1;n++} END{if(n>0) printf "%.1f",s/n; else print "?"}')
                    coldE=$(echo "$debug_lines" | grep -oP 'coldE=\K[-0-9.e+]+'  | awk '{s+=$1;n++} END{if(n>0) printf "%.4f",s/n; else print "?"}')
                    feasR=$(echo "$debug_lines" | grep -oP 'feasible_count=\K[0-9]+' | awk '{s+=$1;n++} END{if(n>0) printf "%.1f",s/n; else print "?"}')
                else
                    rate="?"; distE="?"; coldE="?"; feasR="?"
                fi

                printf "%-6s %-6s %-4s | %-5s %-5s %-6s | %-8s %-8s %-8s %-5s\n" \
                    "$tmax" "$k" "$seed" "$probes" "$feasible" "$Lbest" \
                    "$rate" "$distE" "$coldE" "$feasR"
            done
        done
    done

    echo ""
    echo "sweep_debug complete."

else
    echo "Unknown mode: $MODE  (use: debug | nodebug | sweep)"
    exit 1
fi
