#!/usr/bin/env bash
set -euo pipefail

# Localhost multi-agent twin-node test for Firefly EdgeOS.
# Spawns two instances on different telemetry, WAN, and state paths, but
# shares the local multi-agent UDP port range so the local broadcast layer
# discovers a sibling on the same machine.

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${REPO_ROOT}/target/release/firefly_edgeos"

if [[ ! -x "$BIN" ]]; then
    echo "Release binary not found. Building now..."
    (cd "$REPO_ROOT" && cargo build --release)
fi

mkdir -p \
    "${REPO_ROOT}/wild_workspace/twin_a" \
    "${REPO_ROOT}/wild_workspace/twin_b" \
    "${REPO_ROOT}/wild_workspace/twin_skills_a" \
    "${REPO_ROOT}/wild_workspace/twin_skills_b" \
    "${REPO_ROOT}/wild_workspace/twin_tools_a" \
    "${REPO_ROOT}/wild_workspace/twin_tools_b"

# Kill any previous twin nodes from this script.
for f in wild_workspace/twin_a.pid wild_workspace/twin_b.pid; do
    if [[ -f "${REPO_ROOT}/${f}" ]]; then
        old_pid="$(cat "${REPO_ROOT}/${f}")"
        if kill -0 "$old_pid" 2>/dev/null; then
            echo "Stopping previous twin node PID ${old_pid}"
            kill "$old_pid" 2>/dev/null || true
            sleep 1
        fi
    fi
done

# The default multi-agent secret is hostname-derived, so both nodes share it.
# Setting it explicitly removes any chance of a mismatch.
SHARED_SECRET="twin-localhost-test-secret"

start_node() {
    local node="$1"
    local port_offset="$2"
    local wan_tcp="$3"
    local wan_ws="$4"
    local peers="$5"
    local log_file="$6"

    cd "$REPO_ROOT"
    env \
        "FIREFLY_TELEMETRY_PORT=$((8080 + port_offset))" \
        "FIREFLY_MULTI_AGENT_PORT_START=5001" \
        "FIREFLY_MULTI_AGENT_PORT_END=5010" \
        "FIREFLY_WAN_TCP_PORT=${wan_tcp}" \
        "FIREFLY_WAN_WS_PORT=${wan_ws}" \
        "FIREFLY_STATE_FILE=wild_workspace/twin_state_${node}.json" \
        "FIREFLY_SLED_DB_PATH=wild_workspace/twin_strategy_db_${node}" \
        "FIREFLY_WILD_WORKSPACE_DIR=wild_workspace/twin_${node}" \
        "FIREFLY_METRICS_LOG=wild_workspace/twin_metrics_${node}.jsonl" \
        "FIREFLY_SKILLS_DIR=wild_workspace/twin_skills_${node}" \
        "FIREFLY_TOOLS_DIR=wild_workspace/twin_tools_${node}" \
        "FIREFLY_MULTI_AGENT_SECRET=${SHARED_SECRET}" \
        "FIREFLY_PEERS=${peers}" \
        "$BIN" > "${log_file}" 2>&1 &

    local pid=$!
    echo "$pid" > "${REPO_ROOT}/wild_workspace/twin_${node}.pid"
    echo "  Node ${node} PID=${pid}  telemetry=http://127.0.0.1:$((8080 + port_offset))  log=${log_file}"
}

echo "Starting Firefly EdgeOS twin nodes..."

# Node A listens on WAN TCP 6001; B connects to it.
start_node a 0 6001 6002 "127.0.0.1:6101" "${REPO_ROOT}/wild_workspace/twin_a.log"

# Node B listens on WAN TCP 6101; A connects to it.
start_node b 10 6101 6102 "127.0.0.1:6001" "${REPO_ROOT}/wild_workspace/twin_b.log"

echo ""
echo "Both nodes are running. Give them ~5-10 seconds to negotiate TCP peers,"
echo "then check the logs for:"
echo "  '📡 Broadcast signed engram' (outbound)"
echo "  '📥 [TELEPATHIC EXCHANGER]: Merging signed external engram' (inbound)"
echo ""
echo "Stop with: kill $(cat "${REPO_ROOT}/wild_workspace/twin_a.pid") $(cat "${REPO_ROOT}/wild_workspace/twin_b.pid")"
