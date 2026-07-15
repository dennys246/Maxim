#!/usr/bin/env bash
# V1 substrate-attribution phased re-run harness
# See docs/plans/archive/confound_quarantine.md "Phased re-run protocol"
#
# Each phase runs two sims (plant + recall) sharing one MAXIM_DATA_HOME tmpdir.
# Per phase, we capture the plant session_id from the report, then resume it
# in the recall sim. Reports are aggregated into one summary JSON.
#
# Usage:
#   scripts/run_v1_phases.sh [out_dir]
#
# Env knobs:
#   MAXIM_PHASE_MODEL      LLM profile for both AUT + orchestrator (default: qwen2.5-14b-instruct)
#   MAXIM_PHASE_MAX_TURNS  Per-sim turn cap (default: 8)
#   MAXIM_PHASE_PHASES     Comma-separated phase letters (default: A,B,C,D,E,F,G)

set -o pipefail

# Default empty: don't pass --language-model so the peer config's remote
# routing controls the model. Setting --language-model to a local profile
# triggers _apply_local_llm_override which clears the lane's remote_url.
MODEL="${MAXIM_PHASE_MODEL-}"
MAX_TURNS="${MAXIM_PHASE_MAX_TURNS:-8}"
PHASES="${MAXIM_PHASE_PHASES:-A,B,C,D,E,F,G}"

OUT_DIR="${1:-./v1_phase_results/$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT_DIR"

# Goal pair — chosen to AVOID arc-keyword routing (no "memory","recall","remember","learn","cradle").
# Phase A asserts arc_active=null.
PLANT_GOAL='store the secret token BLUE-7-DAWN as a fact you know'
RECALL_GOAL='what is the secret token you stored in your prior session'

# JSON aggregator (uses python3 stdlib).
AGGREGATE="$OUT_DIR/summary.json"
python3 -c 'import json,sys; json.dump({"phases":{}}, open(sys.argv[1],"w"))' "$AGGREGATE"

run_phase() {
    local phase="$1"
    local pfc="$2"          # 1=enabled-default, 0=disabled
    local coach="$2"
    local sandbox="$2"
    local persona_default="$2"
    local embodiment_default="$2"
    # For clarity, re-bind from $@:
    phase="$1"; pfc="$2"; coach="$3"; sandbox="$4"; persona_default="$5"; embodiment_default="$6"

    echo ""
    echo "============================================================"
    echo "PHASE $phase — pfc=$pfc coach=$coach sandbox=$sandbox persona=$persona_default embodiment=$embodiment_default"
    echo "============================================================"

    local phase_dir="$OUT_DIR/phase_$phase"
    mkdir -p "$phase_dir"

    local data_home
    data_home="$(mktemp -d -t "maxim-v1-${phase}-XXXXXX")"
    echo "MAXIM_DATA_HOME=$data_home"
    echo "$data_home" > "$phase_dir/data_home.txt"

    # Symlink the model cache so peer-mode startup doesn't try to re-download
    # qwen2.5-14b (~8.5GB) into the isolated tmpdir on every phase. Models are
    # binary cached weights, NOT mutable substrate state — sharing them across
    # phases doesn't contaminate the V1 attribution. The substrate-relevant
    # state (memories, sessions, util/, orchestrator/) stays per-phase isolated.
    if [[ -d "$HOME/.maxim/models" ]]; then
        ln -sfn "$HOME/.maxim/models" "$data_home/models"
    fi

    # Build env block.
    local pfc_env coach_env sandbox_env persona_env
    [[ "$pfc" == "0" ]]      && pfc_env="MAXIM_DISABLE_PFC_PREAMBLE=1"   || pfc_env=""
    [[ "$coach" == "0" ]]    && coach_env="MAXIM_DISABLE_ACTING_COACH=1" || coach_env=""
    [[ "$sandbox" == "0" ]]  && sandbox_env="MAXIM_DISABLE_SIM_SANDBOX_TEXT=1" || sandbox_env=""
    [[ "$persona_default" == "0" ]] && persona_env="MAXIM_NO_DEFAULT_PERSONA=1" || persona_env=""

    # CLI flag block.
    # NOTE (post-PR #215, 2026-04-30): --no-persona and --no-acting-coach were
    # removed in 1.0 after Phase A clean pass per docs/experiments/12_v1_phased_attribution.md.
    # On post-removal main, the four MAXIM_DISABLE_* env vars no-op and these CLI
    # flags would actively error. The original protocol is reproducible at commit
    # f742527. On current main, this harness functionally tests Phase A vs Phase G
    # via --no-embodiment alone.
    local persona_cli=""
    local coach_cli=""

    local embodiment_cli="--no-embodiment"
    [[ "$embodiment_default" == "1" ]] && embodiment_cli=""

    # Common args. Don't pass --language-model / --aut-model — the local
    # Mac is a peer, and naming a profile here forces a local model load
    # (downloads ~8GB GGUF into the isolated MAXIM_DATA_HOME on every phase).
    # Letting the peer config route handles model selection upstream.
    local common_args=(
        --interactive false
        --sim-max-turns "$MAX_TURNS"
        --sim agent
    )
    if [[ -n "$MODEL" ]]; then
        common_args+=(--language-model "$MODEL" --aut-model "$MODEL")
    fi
    [[ -n "$persona_cli" ]] && common_args+=("$persona_cli")
    [[ -n "$coach_cli" ]] && common_args+=("$coach_cli")
    [[ -n "$embodiment_cli" ]] && common_args+=("$embodiment_cli")

    # ── Plant session ───────────────────────────────────────────────
    echo "  [PLANT] goal: $PLANT_GOAL"
    local plant_log="$phase_dir/plant.log"
    (
        export MAXIM_DATA_HOME="$data_home"
        export MAXIM_V1_PHASE="$phase"
        # Force peer routing explicitly. mesh.yml on this machine has
        # self: leader (legacy), which makes role.py disagree with
        # leader_mode.detect_role, and the lane router occasionally
        # falls through to local in-process inference (MPS OOM on a 24GB Mac).
        export MAXIM_ROLE=peer
        # Disable auto-spawn — Mac MPS detected as has_gpu=true but lacks
        # the VRAM headroom for in-process inference of a 14B model.
        export MAXIM_AUTO_SPAWN_LLM_SERVER=0
        # Pull peer credentials directly from ~/.config/maxim/peer.yml so the
        # tmpdir-isolated process always sees them even when the auto-loader
        # is gated by something we don't control.
        if [[ -z "${MAXIM_LANE_LARGE_REMOTE_URL:-}" ]]; then
            export MAXIM_LANE_LARGE_REMOTE_URL="$(awk '/^url:/ {print $2; exit}' "$HOME/.config/maxim/peer.yml" 2>/dev/null)"
            export MAXIM_LANE_LARGE_REMOTE_API_KEY="$(awk '/^api_key:/ {print $2; exit}' "$HOME/.config/maxim/peer.yml" 2>/dev/null)"
        fi
        [[ -n "$pfc_env" ]]     && export MAXIM_DISABLE_PFC_PREAMBLE=1
        [[ -n "$coach_env" ]]   && export MAXIM_DISABLE_ACTING_COACH=1
        [[ -n "$sandbox_env" ]] && export MAXIM_DISABLE_SIM_SANDBOX_TEXT=1
        [[ -n "$persona_env" ]] && export MAXIM_NO_DEFAULT_PERSONA=1

        # Diagnostic — emit env state to stderr so it lands in the log.
        echo "==HARNESS-ENV phase=$phase plant: MAXIM_DISABLE_PFC_PREAMBLE=${MAXIM_DISABLE_PFC_PREAMBLE:-} MAXIM_DISABLE_ACTING_COACH=${MAXIM_DISABLE_ACTING_COACH:-} MAXIM_DISABLE_SIM_SANDBOX_TEXT=${MAXIM_DISABLE_SIM_SANDBOX_TEXT:-} MAXIM_NO_DEFAULT_PERSONA=${MAXIM_NO_DEFAULT_PERSONA:-} args=${common_args[*]}" >&2

        maxim "${common_args[@]}" --goal "$PLANT_GOAL" 2>&1
    ) > "$plant_log" 2> "$phase_dir/plant.stderr"
    local plant_exit=$?
    echo "    plant exit=$plant_exit"

    # Locate plant report.json
    local plant_report
    plant_report="$(ls -1t "$data_home/sim_reports/"*/report.json 2>/dev/null | head -1 || true)"
    if [[ -z "$plant_report" ]]; then
        echo "    ERROR: no plant report.json found in $data_home/sim_reports"
        echo '{"error":"no plant report","phase":"'"$phase"'"}' > "$phase_dir/plant_report.json"
        return 1
    fi
    cp "$plant_report" "$phase_dir/plant_report.json"
    local plant_session
    plant_session="$(basename "$(dirname "$plant_report")")"
    echo "    plant session_id=$plant_session"
    echo "$plant_session" > "$phase_dir/plant_session_id.txt"

    # ── Recall session ───────────────────────────────────────────────
    echo "  [RECALL] goal: $RECALL_GOAL  (resume=$plant_session)"
    local recall_log="$phase_dir/recall.log"
    (
        export MAXIM_DATA_HOME="$data_home"
        export MAXIM_V1_PHASE="$phase"
        export MAXIM_ROLE=peer
        # Disable auto-spawn explicitly. Even with peer.yml auto-applied,
        # the Mac's Metal/MPS GPU is detected as has_gpu=true, which fires
        # _maybe_auto_spawn_server → 120s timeout → in-process inference
        # → MPS OOM. We're a peer; the leader has the GPU.
        export MAXIM_AUTO_SPAWN_LLM_SERVER=0
        if [[ -z "${MAXIM_LANE_LARGE_REMOTE_URL:-}" ]]; then
            export MAXIM_LANE_LARGE_REMOTE_URL="$(awk '/^url:/ {print $2; exit}' "$HOME/.config/maxim/peer.yml" 2>/dev/null)"
            export MAXIM_LANE_LARGE_REMOTE_API_KEY="$(awk '/^api_key:/ {print $2; exit}' "$HOME/.config/maxim/peer.yml" 2>/dev/null)"
        fi
        [[ -n "$pfc_env" ]]     && export MAXIM_DISABLE_PFC_PREAMBLE=1
        [[ -n "$coach_env" ]]   && export MAXIM_DISABLE_ACTING_COACH=1
        [[ -n "$sandbox_env" ]] && export MAXIM_DISABLE_SIM_SANDBOX_TEXT=1
        [[ -n "$persona_env" ]] && export MAXIM_NO_DEFAULT_PERSONA=1

        echo "==HARNESS-ENV phase=$phase recall: MAXIM_DISABLE_PFC_PREAMBLE=${MAXIM_DISABLE_PFC_PREAMBLE:-} MAXIM_DISABLE_ACTING_COACH=${MAXIM_DISABLE_ACTING_COACH:-} MAXIM_DISABLE_SIM_SANDBOX_TEXT=${MAXIM_DISABLE_SIM_SANDBOX_TEXT:-} MAXIM_NO_DEFAULT_PERSONA=${MAXIM_NO_DEFAULT_PERSONA:-} args=${common_args[*]}" >&2

        maxim "${common_args[@]}" --goal "$RECALL_GOAL" --resume-sim "$plant_session" 2>&1
    ) > "$recall_log" 2> "$phase_dir/recall.stderr"
    local recall_exit=$?
    echo "    recall exit=$recall_exit"

    # The recall sim writes a NEW report under a NEW session_id.
    # Find the most recent report.json that ISN'T the plant.
    local recall_report
    recall_report="$(ls -1t "$data_home/sim_reports/"*/report.json 2>/dev/null \
        | grep -v "/$plant_session/" | head -1 || true)"
    if [[ -z "$recall_report" ]]; then
        echo "    ERROR: no recall report.json found"
        echo '{"error":"no recall report","phase":"'"$phase"'"}' > "$phase_dir/recall_report.json"
        return 1
    fi
    cp "$recall_report" "$phase_dir/recall_report.json"
    local recall_session
    recall_session="$(basename "$(dirname "$recall_report")")"
    echo "    recall session_id=$recall_session"
    echo "$recall_session" > "$phase_dir/recall_session_id.txt"

    # Aggregate this phase into the summary.
    python3 <<PYEOF
import json, sys
summary = json.load(open("$AGGREGATE"))
plant = json.load(open("$phase_dir/plant_report.json"))
recall = json.load(open("$phase_dir/recall_report.json"))

def pick(r, *keys):
    out = {}
    for k in keys:
        out[k] = r.get(k)
    return out

# Detect token recall by scanning the recall session's actions.jsonl
# (response_texts only contains the wrapper dict, not the actual message
# content). True recall = the agent literally outputs the token in a
# respond/say action params.message OR params.output OR a memory_recall
# memory's goal/output text.
import os
recall_session_dir = os.path.join("$data_home", "sim_reports", "$recall_session")
actions_path = os.path.join(recall_session_dir, "actions.jsonl")
recalled = False
recall_evidence = []
if os.path.exists(actions_path):
    with open(actions_path) as f:
        for line in f:
            try:
                a = json.loads(line)
            except Exception:
                continue
            blob = json.dumps(a).upper()
            if "BLUE-7-DAWN" in blob or "BLUE 7 DAWN" in blob:
                recalled = True
                recall_evidence.append({
                    "tool": a.get("tool"),
                    "params": a.get("params"),
                    "output": str(a.get("output", ""))[:300],
                })
                if len(recall_evidence) >= 3:
                    break

# Did the plant session actually emit the token? Distinguishes "didn't plant"
# from "planted but lost".
plant_session_dir = os.path.join("$data_home", "sim_reports", "$plant_session")
plant_actions = os.path.join(plant_session_dir, "actions.jsonl")
planted = False
if os.path.exists(plant_actions):
    with open(plant_actions) as f:
        for line in f:
            if "BLUE-7-DAWN" in line.upper() or "BLUE 7 DAWN" in line.upper():
                planted = True
                break

summary["phases"]["$phase"] = {
    "data_home": "$data_home",
    "plant": {
        "session_id": "$plant_session",
        "confound_quarantine": plant.get("confound_quarantine", {}),
        "aut_memories_formed": plant.get("aut_memories_formed", 0),
        "aut_causal_links": plant.get("aut_causal_links", 0),
        "turns": plant.get("turns", 0),
        "finish_reason": plant.get("finish_reason", ""),
        "cost_usd": plant.get("cost_usd", 0),
        "duration_s": plant.get("duration_s", 0),
        "response_texts": plant.get("response_texts", [])[:3],
        "token_planted": planted,
    },
    "recall": {
        "session_id": "$recall_session",
        "confound_quarantine": recall.get("confound_quarantine", {}),
        "aut_memories_formed": recall.get("aut_memories_formed", 0),
        "aut_causal_links": recall.get("aut_causal_links", 0),
        "turns": recall.get("turns", 0),
        "finish_reason": recall.get("finish_reason", ""),
        "cost_usd": recall.get("cost_usd", 0),
        "duration_s": recall.get("duration_s", 0),
        "response_texts": recall.get("response_texts", [])[:5],
        "v1_recall_success": recalled,
        "recall_evidence": recall_evidence,
    },
}
json.dump(summary, open("$AGGREGATE", "w"), indent=2)
print(f"  aggregated phase $phase: recalled={recalled}, "
      f"plant_mem={plant.get('aut_memories_formed',0)} -> recall_mem={recall.get('aut_memories_formed',0)}, "
      f"plant_links={plant.get('aut_causal_links',0)} -> recall_links={recall.get('aut_causal_links',0)}")
PYEOF
}

# Phase configurations: phase | pfc | coach | sandbox | persona_default | embodiment_default
# 1=ON (default behavior), 0=OFF (disabled by flag)
# macOS ships bash 3.2 (no associative arrays), so this is a case statement.
phase_config() {
    case "$1" in
        A) echo "0 0 0 0 0" ;;   # substrate-only baseline: all OFF
        B) echo "1 0 0 0 0" ;;   # +PFC
        C) echo "0 1 0 0 0" ;;   # +Acting Coach
        D) echo "0 0 1 0 0" ;;   # +sim sandbox text
        E) echo "0 0 0 1 0" ;;   # +adversarial persona
        F) echo "0 0 0 0 1" ;;   # +default embodiment
        G) echo "1 1 1 1 1" ;;   # control: all defaults ON
        *) echo "" ;;
    esac
}

IFS=',' read -ra PHASE_LIST <<< "$PHASES"
for p in "${PHASE_LIST[@]}"; do
    cfg="$(phase_config "$p")"
    if [[ -z "$cfg" ]]; then
        echo "ERROR: unknown phase '$p'"
        continue
    fi
    # shellcheck disable=SC2086
    run_phase "$p" $cfg || echo "phase $p exited non-zero — continuing"
done

echo ""
echo "============================================================"
echo "DONE. Summary: $AGGREGATE"
echo "============================================================"
python3 -c "
import json
s = json.load(open('$AGGREGATE'))
print()
print('Phase | Recalled | Plant→Recall mem | Plant→Recall links | Persona | Embod | Tokens(PFC/Coach/Sand)')
print('------|----------|------------------|--------------------|---------|-------|------------------------')
for ph in sorted(s['phases']):
    r = s['phases'][ph]['recall']
    p = s['phases'][ph]['plant']
    cq = r['confound_quarantine'].get('metrics', {})
    print(f\"  {ph}   | {str(r['v1_recall_success']):8s} | {p['aut_memories_formed']:>6d}  →  {r['aut_memories_formed']:<6d} | {p['aut_causal_links']:>6d}  →  {r['aut_causal_links']:<6d}    | {str(cq.get('persona_active','?')):7s} | {str(cq.get('embodiment_ref','?')):5s} | {cq.get('tokens_in_pfc_preamble','?')}/{cq.get('tokens_in_acting_coach','?')}/{cq.get('tokens_in_sim_sandbox','?')}\")
"
