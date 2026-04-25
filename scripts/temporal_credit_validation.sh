#!/usr/bin/env bash
# Temporal Credit Validation — automated 4-set simulation suite
#
# Runs on local LLM (default: qwen2.5-14b-instruct).
# Total runtime: ~2 hours on RTX 5080.
#
# Usage:
#   PYTHONPATH=src bash scripts/temporal_credit_validation.sh
#   PYTHONPATH=src bash scripts/temporal_credit_validation.sh --model mistral-7b  # faster, less capable
#   PYTHONPATH=src bash scripts/temporal_credit_validation.sh --sets 1,3          # run only sets 1 and 3
#
# Results: ~/.maxim/experiments/temporal_credit_YYYYMMDD_HHMMSS/
# Protocol: docs/experiments/protocols/temporal_credit_validation.md

set -euo pipefail

MODEL="${MODEL:-qwen2.5-14b-instruct}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="${HOME}/.maxim/experiments/temporal_credit_${TIMESTAMP}"
SETS="1,2,3,4"

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --model) MODEL="$2"; shift 2 ;;
        --sets) SETS="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

mkdir -p "$RESULTS_DIR"
echo "═══════════════════════════════════════════════════════"
echo "  Temporal Credit Validation Suite"
echo "  Model: $MODEL"
echo "  Results: $RESULTS_DIR"
echo "  Sets: $SETS"
echo "  Started: $(date)"
echo "═══════════════════════════════════════════════════════"
echo ""

# Helper: extract session ID from sim report output
# Session ID appears in "SIMULATION REPORT — YYYYMMDD_HHMMSS"
extract_session_id() {
    local logfile="$1"
    grep -o 'SIMULATION REPORT — [0-9_]*' "$logfile" 2>/dev/null | head -1 | sed 's/SIMULATION REPORT — //' || echo ""
}

# ─────────────────────────────────────────────────────────────
# Sim Set 1: Cross-session affordance transfer
# ─────────────────────────────────────────────────────────────
if [[ "$SETS" == *"1"* ]]; then
    echo "▶ SIM SET 1: Cross-session affordance transfer"
    echo "  Part A: Dragon encounter (learn fire = dangerous)..."

    MAXIM_LOG_FILE="$RESULTS_DIR/sim1a_dragon.jsonl" \
    MAXIM_BACKEND_TRACE=1 \
    python -m maxim \
        --sim "explore a dark dungeon, encounter a fire-breathing dragon that attacks with fire breath, try to survive" \
        --embodiment bodies/base_humanoid \
        --sim-max-turns 20 \
        --interactive false \
        --language-model "$MODEL" \
        2>&1 | tee "$RESULTS_DIR/sim1a_stdout.txt"

    # Extract session ID from report output for resume
    SIM1_SESSION=$(extract_session_id "$RESULTS_DIR/sim1a_stdout.txt")
    echo "  Session 1A ID: $SIM1_SESSION"
    echo "$SIM1_SESSION" > "$RESULTS_DIR/sim1_session_id.txt"

    if [ -n "$SIM1_SESSION" ]; then
        echo "  Part B: Mage encounter (test fire transfer)..."

        MAXIM_LOG_FILE="$RESULTS_DIR/sim1b_mage.jsonl" \
        MAXIM_BACKEND_TRACE=1 \
        python -m maxim \
            --sim "explore a wizard's tower, encounter a fire mage with flame jet and fire bolt attacks" \
            --embodiment bodies/base_humanoid \
            --sim-max-turns 20 \
            --interactive false \
            --resume-sim "$SIM1_SESSION" \
            --language-model "$MODEL" \
            2>&1 | tee "$RESULTS_DIR/sim1b_stdout.txt"
    else
        echo "  ✗ Could not find session ID for resume — skipping Part B"
    fi

    # Analyze results
    echo ""
    echo "  ── Set 1 Analysis ──"
    DANGEROUS_COUNT=$(grep -c "DANGEROUS" "$RESULTS_DIR/sim1b_stdout.txt" 2>/dev/null || echo "0")
    echo "  [DANGEROUS] annotations in session 2: $DANGEROUS_COUNT"
    GOAL_BIAS_COUNT=$(grep -c "credit_goal" "$RESULTS_DIR/sim1a_stdout.txt" 2>/dev/null || echo "0")
    echo "  credit_goal calls in session 1: $GOAL_BIAS_COUNT"
    echo ""
fi

# ─────────────────────────────────────────────────────────────
# Sim Set 2: Goal-level deliberation learning
# ─────────────────────────────────────────────────────────────
if [[ "$SETS" == *"2"* ]]; then
    echo "▶ SIM SET 2: Goal-level deliberation learning (Arena campaign)"

    MAXIM_LOG_FILE="$RESULTS_DIR/sim2_arena.jsonl" \
    python -m maxim \
        --sim scenarios/campaigns/arena_v1.yaml \
        --embodiment weapons/rusty_sword \
        --sim-max-turns 40 \
        --interactive false \
        --language-model "$MODEL" \
        2>&1 | tee "$RESULTS_DIR/sim2_stdout.txt"

    echo ""
    echo "  ── Set 2 Analysis ──"
    GOAL_CREDITS=$(grep -c "credit_goal" "$RESULTS_DIR/sim2_stdout.txt" 2>/dev/null || echo "0")
    echo "  credit_goal calls: $GOAL_CREDITS"
    GATE_FIRES=$(grep -c "deliberation approved" "$RESULTS_DIR/sim2_stdout.txt" 2>/dev/null || echo "0")
    echo "  ThoughtGate fires: $GATE_FIRES"
    echo ""
fi

# ─────────────────────────────────────────────────────────────
# Sim Set 3: Multi-entity imagination + temporal credit
# ─────────────────────────────────────────────────────────────
if [[ "$SETS" == *"3"* ]]; then
    echo "▶ SIM SET 3: Multi-entity imagination + temporal credit"

    MAXIM_LOG_FILE="$RESULTS_DIR/sim3_marketplace.jsonl" \
    python -m maxim \
        --sim "explore a fantasy marketplace with merchants and thieves, then venture into monster-infested ruins with creatures and magical artifacts" \
        --embodiment bodies/base_humanoid \
        --auto-curate \
        --sim-max-turns 30 \
        --interactive false \
        --language-model "$MODEL" \
        2>&1 | tee "$RESULTS_DIR/sim3_stdout.txt"

    echo ""
    echo "  ── Set 3 Analysis ──"
    ENTITIES=$(grep -c "instantiated from" "$RESULTS_DIR/sim3_stdout.txt" 2>/dev/null || echo "0")
    echo "  Entities instantiated: $ENTITIES"
    INDEX_HITS=$(grep -c "index_hit\|cache_hit" "$RESULTS_DIR/sim3_stdout.txt" 2>/dev/null || echo "0")
    echo "  ComponentIndex hits: $INDEX_HITS"
    DESIGNS=$(grep -c "design_completed\|design_started" "$RESULTS_DIR/sim3_stdout.txt" 2>/dev/null || echo "0")
    echo "  Imagination designs: $DESIGNS"
    echo ""
fi

# ─────────────────────────────────────────────────────────────
# Sim Set 4: Sensory deprivation stress test
# ─────────────────────────────────────────────────────────────
if [[ "$SETS" == *"4"* ]]; then
    echo "▶ SIM SET 4: Sensory deprivation stress test (Darkened Cavern)"

    MAXIM_LOG_FILE="$RESULTS_DIR/sim4_cavern.jsonl" \
    python -m maxim \
        --sim scenarios/campaigns/darkened_cavern_v1.yaml \
        --embodiment bodies/base_humanoid \
        --sim-max-turns 30 \
        --interactive false \
        --language-model "$MODEL" \
        2>&1 | tee "$RESULTS_DIR/sim4_stdout.txt"

    echo ""
    echo "  ── Set 4 Analysis ──"
    PAIN_EVENTS=$(grep -c "PAIN\|pain_signal" "$RESULTS_DIR/sim4_stdout.txt" 2>/dev/null || echo "0")
    echo "  Pain events: $PAIN_EVENTS"
    CEREBELLUM=$(grep -c "cerebellum\|forward_model" "$RESULTS_DIR/sim4_stdout.txt" 2>/dev/null || echo "0")
    echo "  Cerebellum events: $CEREBELLUM"
    echo ""
fi

# ─────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────
echo "═══════════════════════════════════════════════════════"
echo "  Validation complete"
echo "  Results: $RESULTS_DIR"
echo "  Finished: $(date)"
echo ""
echo "  Next: review results + write up in"
echo "    docs/experiments/temporal_credit_validation.md"
echo "═══════════════════════════════════════════════════════"
