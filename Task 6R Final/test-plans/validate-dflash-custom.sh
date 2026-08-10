#!/usr/bin/env bash
# =============================================================================
# Task 6R: Custom DFlash Replay Validation Script
# =============================================================================
#
# This script validates the Task 6R custom DFlash replay implementation by:
# 1. Starting llama-server with --beefix-dflash-custom enabled
# 2. Sending inference requests that trigger speculative decoding
# 3. Capturing and analyzing diagnostic logs (with reason codes R0-R9)
# 4. Reporting which replay path was taken and how many times
# 5. Identifying skipped operations and their reasons
#
# Usage:
#   ./validate-dflash-custom.sh [OPTIONS]
#
# Required Environment Variables:
#   TARGET_MODEL  - Path to the target GGUF model file
#   DRAFT_MODEL   - Path to the DFlash draft GGUF model file
#
# Optional Environment Variables:
#   SERVER_PORT   - Port for llama-server (default: 18080)
#   N_DRAFT_MAX   - Max draft tokens (default: 8)
#   N_REQUESTS    - Number of test requests to send (default: 10)
#   MAX_TOKENS    - Max output tokens per request (default: 64)
#   LLAMA_SERVER  - Path to llama-server binary (default: ./build/bin/llama-server)
#   LOG_FILE      - Path to server log file (default: ./dflash-test.log)
#   TIMEOUT       - Server startup timeout in seconds (default: 120)
#
# Example:
#   export TARGET_MODEL=/models/qwen3-27b-dflash-v2.gguf
#   export DRAFT_MODEL=/models/qwen3-27b-dflash-v2-dflash.gguf
#   ./validate-dflash-custom.sh
#
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVER_PORT="${SERVER_PORT:-18080}"
N_DRAFT_MAX="${N_DRAFT_MAX:-8}"
N_REQUESTS="${N_REQUESTS:-10}"
MAX_TOKENS="${MAX_TOKENS:-64}"
LLAMA_SERVER="${LLAMA_SERVER:-./build/bin/llama-server}"
LOG_FILE="${LOG_FILE:-${SCRIPT_DIR}/dflash-test.log}"
TIMEOUT="${TIMEOUT:-120}"
SERVER_PID=""

# Colors for terminal output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

log_info() {
    echo -e "${BLUE}[INFO]${NC} $*"
}

log_success() {
    echo -e "${GREEN}[PASS]${NC} $*"
}

log_warning() {
    echo -e "${YELLOW}[WARN]${NC} $*"
}

log_error() {
    echo -e "${RED}[FAIL]${NC} $*"
}

log_header() {
    echo ""
    echo -e "${BOLD}${CYAN}========================================${NC}"
    echo -e "${BOLD}${CYAN} $*${NC}"
    echo -e "${BOLD}${CYAN}========================================${NC}"
    echo ""
}

cleanup() {
    if [[ -n "$SERVER_PID" ]]; then
        log_info "Stopping llama-server (PID: $SERVER_PID)..."
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}

trap cleanup EXIT

# ---------------------------------------------------------------------------
# Pre-flight Checks
# ---------------------------------------------------------------------------

preflight_checks() {
    log_header "PRE-FLIGHT CHECKS"

    # Check required environment variables
    if [[ -z "${TARGET_MODEL:-}" ]]; then
        log_error "TARGET_MODEL environment variable not set."
        exit 1
    fi

    if [[ -z "${DRAFT_MODEL:-}" ]]; then
        log_error "DRAFT_MODEL environment variable not set."
        exit 1
    fi

    # Check model files exist
    if [[ ! -f "$TARGET_MODEL" ]]; then
        log_error "Target model not found: $TARGET_MODEL"
        exit 1
    fi
    log_success "Target model found: $TARGET_MODEL"

    if [[ ! -f "$DRAFT_MODEL" ]]; then
        log_error "Draft model not found: $DRAFT_MODEL"
        exit 1
    fi
    log_success "Draft model found: $DRAFT_MODEL"

    # Check llama-server binary
    if [[ ! -x "$LLAMA_SERVER" ]]; then
        log_error "llama-server binary not found or not executable: $LLAMA_SERVER"
        log_info "Run 'cmake --build build --config Release' first."
        exit 1
    fi
    log_success "llama-server binary found: $LLAMA_SERVER"

    # Check for curl
    if ! command -v curl &> /dev/null; then
        log_error "curl is required but not installed."
        exit 1
    fi

    # Check port availability
    if command -v netstat &> /dev/null; then
        if netstat -tuln 2>/dev/null | grep -q ":${SERVER_PORT} "; then
            log_warning "Port $SERVER_PORT is already in use."
            log_info "Attempting to use port $((SERVER_PORT + 1)) instead."
            SERVER_PORT=$((SERVER_PORT + 1))
        fi
    fi

    log_info "Server will listen on port: $SERVER_PORT"
    log_info "Log file: $LOG_FILE"
}

# ---------------------------------------------------------------------------
# Step 1: Start llama-server with Custom DFlash
# ---------------------------------------------------------------------------

start_server() {
    log_header "STEP 1: START SERVER"

    log_info "Starting llama-server with --beefix-dflash-custom..."
    echo ""

    # Clear previous log
    > "$LOG_FILE"

    # Start server in background
    "$LLAMA_SERVER" \
        -m "$TARGET_MODEL" \
        --spec-type draft-dflash \
        --spec-draft-model "$DRAFT_MODEL" \
        --spec-draft-n-max "$N_DRAFT_MAX" \
        --beefix-dflash-custom \
        --trace 1 \
        --port "$SERVER_PORT" \
        --ctx-size 8192 \
        2>&1 | tee "$LOG_FILE" &
    SERVER_PID=$!

    log_info "Server started with PID: $SERVER_PID"
    log_info "Waiting for server to be ready (timeout: ${TIMEOUT}s)..."

    # Wait for server to be ready
    local elapsed=0
    while [[ $elapsed -lt $TIMEOUT ]]; do
        if curl -s "http://localhost:${SERVER_PORT}/health" > /dev/null 2>&1; then
            log_success "Server is ready after ${elapsed}s."
            break
        fi
        # Also check if the log contains the ready message
        if grep -q "listening on" "$LOG_FILE" 2>/dev/null; then
            log_success "Server is ready (detected from log) after ${elapsed}s."
            break
        fi
        sleep 2
        elapsed=$((elapsed + 2))
    done

    if [[ $elapsed -ge $TIMEOUT ]]; then
        log_error "Server failed to start within ${TIMEOUT}s."
        cat "$LOG_FILE"
        exit 1
    fi

    # Verify custom mode was initialized
    if grep -q "\[dflash-custom\] Initialized:" "$LOG_FILE"; then
        log_success "Custom DFlash mode initialized successfully."
        grep "\[dflash-custom\] Initialized:" "$LOG_FILE" | tail -1
    else
        log_warning "Custom DFlash initialization message not found in log."
        log_info "This may indicate tape allocation failed or the model is not a DFlash model."
    fi

    # Display tape allocation info
    if grep -q "\[dflash-custom\] GPU tape allocated:" "$LOG_FILE"; then
        grep "\[dflash-custom\] GPU tape allocated:" "$LOG_FILE" | tail -1 | sed 's/^/  /'
    fi
}

# ---------------------------------------------------------------------------
# Step 2: Send Test Requests
# ---------------------------------------------------------------------------

send_requests() {
    log_header "STEP 2: SEND TEST REQUESTS"

    log_info "Sending $N_REQUESTS test requests..."
    echo ""

    # Test prompts designed to trigger speculative decoding
    local prompts=(
        "Explain how speculative decoding works in large language models."
        "What is the difference between attention and recurrent neural networks?"
        "Write a short Python function that calculates the Fibonacci sequence."
        "Describe the process of quantization in neural network inference."
        "How does KV cache compression affect inference performance?"
        "Explain the concept of flash attention and its benefits."
        "What are the key components of a transformer decoder architecture?"
        "Describe how gradient checkpointing saves memory during training."
        "What is the purpose of layer normalization in deep learning?"
        "Explain the difference between batch inference and streaming inference."
    )

    local success_count=0
    local fail_count=0

    for ((i = 0; i < N_REQUESTS; i++)); do
        local prompt_idx=$((i % ${#prompts[@]}))
        local prompt="${prompts[$prompt_idx]}"

        log_info "Request $((i + 1))/$N_REQUESTS: \"${prompt:0:50}...\""

        local response
        response=$(curl -s -w "\n%{http_code}" \
            "http://localhost:${SERVER_PORT}/v1/chat/completions" \
            -H "Content-Type: application/json" \
            -d "{
                \"model\": \"test\",
                \"messages\": [
                    {\"role\": \"user\", \"content\": \"${prompt}\"}
                ],
                \"temperature\": 0.7,
                \"max_tokens\": ${MAX_TOKENS},
                \"stream\": false
            }" 2>/dev/null)

        local http_code
        http_code=$(echo "$response" | tail -1)
        local body
        body=$(echo "$response" | head -n -1)

        if [[ "$http_code" == "200" ]]; then
            # Extract the generated text for verification
            local gen_text
            gen_text=$(echo "$body" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    print(data['choices'][0]['message']['content'][:80])
except:
    print('(parse error)')
" 2>/dev/null || echo "(extract error)")

            log_success "Request $((i + 1)): HTTP 200 - \"${gen_text}...\""
            success_count=$((success_count + 1))
        else
            log_error "Request $((i + 1)): HTTP $http_code"
            fail_count=$((fail_count + 1))
        fi
    done

    echo ""
    log_info "Request summary: $success_count succeeded, $fail_count failed out of $N_REQUESTS"
}

# ---------------------------------------------------------------------------
# Step 3: Analyze Logs
# ---------------------------------------------------------------------------

analyze_logs() {
    log_header "STEP 3: LOG ANALYSIS"

    echo ""
    log_info "Full log saved to: $LOG_FILE"
    echo ""

    # --- Replay Success Count ---
    local replay_success
    replay_success=$(grep -c "\[dflash-custom\] replay success:" "$LOG_FILE" 2>/dev/null || echo "0")
    echo -e "${BOLD}Replay Successes:${NC} $replay_success"
    if [[ $replay_success -gt 0 ]]; then
        log_success "Custom replay executed successfully $replay_success time(s)."
    else
        log_warning "No replay success messages found."
    fi
    echo ""

    # --- Replay Execute Details ---
    echo -e "${BOLD}Replay Execute Details:${NC}"
    grep "\[dflash-custom\] replay execute:" "$LOG_FILE" 2>/dev/null | tail -5 | sed 's/^/  /' || echo "  (none)"
    echo ""

    # --- Conv Rebuild Count ---
    local conv_rebuild
    conv_rebuild=$(grep -c "\[dflash-custom\] conv rebuild:" "$LOG_FILE" 2>/dev/null || echo "0")
    echo -e "${BOLD}Conv Rebuilds:${NC} $conv_rebuild"
    if [[ $conv_rebuild -gt 0 ]]; then
        log_success "Convolution state rebuilt $conv_rebuild time(s)."
    fi
    echo ""

    # --- Reason Code Analysis ---
    echo -e "${BOLD}Reason Code Summary:${NC}"
    echo ""

    # Extract all reason codes and count them
    local reason_codes
    reason_codes=$(grep -oP "skipped \(\K[R0-9]+" "$LOG_FILE" 2>/dev/null || true)

    if [[ -n "$reason_codes" ]]; then
        echo "$reason_codes" | sort | uniq -c | sort -rn | while read -r count code; do
            local description=""
            case "$code" in
                R0) description="Backup cells insufficient" ;;
                R1) description="Invalid preconditions (null state/ctx or n_accepted<=0)" ;;
                R2) description="Custom mode not enabled" ;;
                R3) description="n_accepted > tokens_captured" ;;
                R4) description="No tape allocated" ;;
                R5) description="No recurrent memory" ;;
                R6) description="No recurrent memory component" ;;
                R7) description="n_backup_cells=0" ;;
                R8) description="No scheduler available" ;;
                R9) description="Conv rebuild skipped (no conv)" ;;
                *)  description="Unknown reason code" ;;
            esac
            if [[ $count -gt 5 ]]; then
                echo -e "  ${RED}${code}${NC} (${count}x) - ${description}"
            elif [[ $count -gt 0 ]]; then
                echo -e "  ${YELLOW}${code}${NC} (${count}x) - ${description}"
            fi
        done
    else
        log_success "No skip/reason code messages detected. All replay attempts succeeded."
    fi
    echo ""

    # --- Fallback Detection ---
    local fallback_count
    fallback_count=$(grep -c "falling back to checkpoint" "$LOG_FILE" 2>/dev/null || echo "0")
    echo -e "${BOLD}Checkpoint Fallbacks:${NC} $fallback_count"
    if [[ $fallback_count -gt 0 ]]; then
        log_warning "Custom replay fell back to checkpoint $fallback_count time(s)."
        echo ""
        echo "  Last fallback messages:"
        grep "falling back to checkpoint" "$LOG_FILE" 2>/dev/null | tail -3 | sed 's/^/    /'
    else
        log_success "No checkpoint fallbacks detected. Custom replay handled all cycles."
    fi
    echo ""

    # --- Permanent Disable Check ---
    local disable_count
    disable_count=$(grep -c "permanently disabled" "$LOG_FILE" 2>/dev/null || echo "0")
    echo -e "${BOLD}Permanent Disables:${NC} $disable_count"
    if [[ $disable_count -gt 0 ]]; then
        log_error "Custom replay was permanently disabled after consecutive failures."
    else
        log_success "Custom replay remained active throughout the test."
    fi
    echo ""

    # --- Failure Count Analysis ---
    local fail_log_count
    fail_log_count=$(grep -c "fail_count" "$LOG_FILE" 2>/dev/null || echo "0")
    if [[ $fail_log_count -gt 0 ]]; then
        echo -e "${BOLD}Failure Counter Events:${NC} $fail_log_count"
        grep "fail_count" "$LOG_FILE" 2>/dev/null | tail -5 | sed 's/^/  /'
    fi
    echo ""

    # --- Backup/Restore Analysis ---
    local backup_skip_count
    backup_skip_count=$(grep -c "backup skipped (R0)" "$LOG_FILE" 2>/dev/null || echo "0")
    echo -e "${BOLD}Backup Skip (R0) Count:${NC} $backup_skip_count"
    if [[ $backup_skip_count -gt 0 ]]; then
        log_error "Backup cells were insufficient during $backup_skip_count operation(s)."
        echo ""
        echo "  Backup skip details:"
        grep "backup skipped (R0)" "$LOG_FILE" 2>/dev/null | tail -3 | sed 's/^/    /'
    else
        log_success "No backup failures detected."
    fi
    echo ""

    # --- Replay Path Summary ---
    echo -e "${BOLD}Replay Path Summary:${NC}"
    echo ""
    echo "  Custom replay successes:    $replay_success"
    echo "  Checkpoint fallbacks:       $fallback_count"
    echo "  Conv rebuilds:              $conv_rebuild"
    echo "  Backup failures (R0):       $backup_skip_count"
    echo "  Permanent disables:         $disable_count"
    echo ""

    local total_replay_attempts=$((replay_success + fallback_count))
    if [[ $total_replay_attempts -gt 0 ]]; then
        local success_rate
        success_rate=$((replay_success * 100 / total_replay_attempts))
        echo -e "  ${BOLD}Custom replay success rate: ${success_rate}%${NC}"
        echo ""
        if [[ $success_rate -ge 90 ]]; then
            log_success "Excellent: Custom replay success rate >= 90%."
        elif [[ $success_rate -ge 50 ]]; then
            log_warning "Moderate: Custom replay success rate between 50-90%. Investigate failures."
        else
            log_error "Poor: Custom replay success rate < 50%. Significant issues detected."
        fi
    fi
}

# ---------------------------------------------------------------------------
# Step 4: Display Key Log Lines
# ---------------------------------------------------------------------------

show_log_samples() {
    log_header "STEP 4: KEY LOG SAMPLES"

    echo -e "${BOLD}Initialization Logs:${NC}"
    grep "\[dflash-custom\]" "$LOG_FILE" 2>/dev/null | head -5 | sed 's/^/  /'
    echo ""

    echo -e "${BOLD}Last 10 Custom DFlash Messages:${NC}"
    grep "\[dflash-custom\]" "$LOG_FILE" 2>/dev/null | tail -10 | sed 's/^/  /'
    echo ""

    echo -e "${BOLD}Warning/Error Messages:${NC}"
    grep -i "warn\|error\|fail" "$LOG_FILE" 2>/dev/null | grep -i "dflash\|specul\|replay" | tail -5 | sed 's/^/  /' || echo "  (none)"
    echo ""
}

# ---------------------------------------------------------------------------
# Step 5: Final Report
# ---------------------------------------------------------------------------

final_report() {
    log_header "FINAL REPORT"

    local replay_success
    replay_success=$(grep -c "\[dflash-custom\] replay success:" "$LOG_FILE" 2>/dev/null || echo "0")
    local fallback_count
    fallback_count=$(grep -c "falling back to checkpoint" "$LOG_FILE" 2>/dev/null || echo "0")
    local disable_count
    disable_count=$(grep -c "permanently disabled" "$LOG_FILE" 2>/dev/null || echo "0")
    local backup_skip_count
    backup_skip_count=$(grep -c "backup skipped (R0)" "$LOG_FILE" 2>/dev/null || echo "0")

    echo ""
    echo -e "${BOLD}Test Configuration:${NC}"
    echo "  Target model:     $TARGET_MODEL"
    echo "  Draft model:      $DRAFT_MODEL"
    echo "  Server port:      $SERVER_PORT"
    echo "  Max draft tokens: $N_DRAFT_MAX"
    echo "  Test requests:    $N_REQUESTS"
    echo "  Max output tokens: $MAX_TOKENS"
    echo ""

    echo -e "${BOLD}Results:${NC}"
    echo ""

    # Overall pass/fail
    local overall_pass=true

    if [[ $replay_success -eq 0 ]]; then
        log_error "No successful custom replay cycles detected."
        overall_pass=false
    else
        log_success "Custom replay executed successfully."
    fi

    if [[ $backup_skip_count -gt 0 ]]; then
        log_error "Backup cell failures detected (R0)."
        overall_pass=false
    else
        log_success "No backup cell failures."
    fi

    if [[ $disable_count -gt 0 ]]; then
        log_error "Custom mode was permanently disabled."
        overall_pass=false
    else
        log_success "Custom mode remained active."
    fi

    if [[ $fallback_count -gt $((N_REQUESTS / 2)) ]]; then
        log_warning "Excessive checkpoint fallbacks ($fallback_count > 50% of requests)."
    fi

    echo ""
    if [[ "$overall_pass" == "true" ]]; then
        echo -e "${GREEN}${BOLD}============================================${NC}"
        echo -e "${GREEN}${BOLD}  OVERALL RESULT: PASS${NC}"
        echo -e "${GREEN}${BOLD}============================================${NC}"
    else
        echo -e "${RED}${BOLD}============================================${NC}"
        echo -e "${RED}${BOLD}  OVERALL RESULT: FAIL${NC}"
        echo -e "${RED}${BOLD}============================================${NC}"
        echo ""
        echo "Review the log analysis above for specific failures."
        echo "Full log: $LOG_FILE"
    fi
    echo ""
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

main() {
    echo ""
    echo -e "${BOLD}${CYAN}============================================${NC}"
    echo -e "${BOLD}${CYAN}  Task 6R: Custom DFlash Replay Validation${NC}"
    echo -e "${BOLD}${CYAN}============================================${NC}"
    echo ""

    preflight_checks
    start_server
    send_requests
    analyze_logs
    show_log_samples
    final_report

    log_info "Test complete. Server log: $LOG_FILE"
    log_info "Server PID: $SERVER_PID (will be stopped on exit)"
}

main "$@"
