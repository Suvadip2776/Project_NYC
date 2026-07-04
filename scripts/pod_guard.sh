#!/usr/bin/env bash
# =============================================================================
# pod_guard.sh — auto-terminate a RunPod pod when a job finishes.
# =============================================================================
# Purpose: kill the "left a GPU pod running overnight" budget leak. Once armed,
# the pod self-terminates on script exit — whether the job SUCCEEDS, ERRORS, or
# is interrupted (Ctrl-C / SIGTERM) — after a short, cancellable grace period.
#
# Cost model reminder: RunPod GPU pods bill by the second while they exist.
# `remove` deletes the pod (container disk gone; a mounted NETWORK VOLUME
# persists). `stop` leaves the pod stopped (no GPU charge, but disk still bills).
# For a shared budget, `remove` + a network volume for /workspace is cheapest.
#
# -----------------------------------------------------------------------------
# Usage (inside a run script executed ON a RunPod pod):
#
#   source "$(dirname "$0")/pod_guard.sh"
#   arm_pod_guard            # registers the auto-terminate trap
#   ...                      # your GPU job here
#                            # -> pod self-terminates when the script exits
#
# Safety: if $RUNPOD_POD_ID is not set (e.g. you're on your Mac, or in an
# interactive shell that isn't a pod), arm_pod_guard is a NO-OP. So sourcing
# this file locally can never nuke anything.
#
# -----------------------------------------------------------------------------
# Config via env vars (set before calling arm_pod_guard):
#   POD_GUARD_MODE=remove|stop   default: remove
#   POD_GUARD_GRACE=<seconds>    default: 60   (Ctrl-C during grace = cancel)
#   POD_GUARD_DISABLE=1          force no-op (dry runs / debugging on a pod)
# =============================================================================

_pod_guard_do_terminate() {
    local pod_id="$1" mode="$2"
    # Prefer runpodctl (preinstalled + self-authorized inside official pods).
    if command -v runpodctl >/dev/null 2>&1; then
        echo "[pod_guard] runpodctl ${mode} pod ${pod_id}"
        runpodctl "${mode}" pod "${pod_id}" && return 0
        echo "[pod_guard] runpodctl failed; trying API fallback..."
    fi
    # Fallback: RunPod GraphQL API (needs RUNPOD_API_KEY in the pod env).
    if [ -n "${RUNPOD_API_KEY:-}" ]; then
        local gql
        if [ "${mode}" = "stop" ]; then
            gql="mutation { podStop(input: {podId: \"${pod_id}\"}) { id desiredStatus } }"
        else
            gql="mutation { podTerminate(input: {podId: \"${pod_id}\"}) }"
        fi
        echo "[pod_guard] calling RunPod GraphQL API (${mode})"
        curl -s -X POST "https://api.runpod.io/graphql?api_key=${RUNPOD_API_KEY}" \
            -H 'Content-Type: application/json' \
            -d "{\"query\": \"$(printf '%s' "$gql" | sed 's/"/\\"/g')\"}" \
            && return 0
    fi
    # Nothing worked — make the failure LOUD so a human terminates it.
    echo "==============================================================" >&2
    echo "[pod_guard] !! COULD NOT SELF-TERMINATE POD ${pod_id} !!"        >&2
    echo "[pod_guard] !! TERMINATE IT MANUALLY IN THE RUNPOD CONSOLE   !!" >&2
    echo "[pod_guard] (install runpodctl, or set RUNPOD_API_KEY)"          >&2
    echo "==============================================================" >&2
    return 1
}

_pod_guard_trap() {
    local rc=$?   # capture the job's exit code first
    local mode="${POD_GUARD_MODE:-remove}"
    local grace="${POD_GUARD_GRACE:-60}"

    echo ""
    echo "[pod_guard] job exited with code ${rc}"
    if [ "${grace}" -gt 0 ] 2>/dev/null; then
        echo "[pod_guard] pod will self-${mode} in ${grace}s — press Ctrl-C to CANCEL and keep it alive."
        # A cancel during the grace window aborts termination (pod stays up).
        if ! sleep "${grace}"; then
            echo "[pod_guard] grace interrupted — termination CANCELLED, pod stays alive."
            return 0
        fi
    fi
    _pod_guard_do_terminate "${RUNPOD_POD_ID}" "${mode}"
}

arm_pod_guard() {
    if [ "${POD_GUARD_DISABLE:-0}" = "1" ]; then
        echo "[pod_guard] disabled (POD_GUARD_DISABLE=1) — pod will NOT auto-terminate."
        return 0
    fi
    if [ -z "${RUNPOD_POD_ID:-}" ]; then
        echo "[pod_guard] not inside a RunPod pod (\$RUNPOD_POD_ID unset) — no-op."
        return 0
    fi
    trap _pod_guard_trap EXIT
    echo "[pod_guard] armed: pod ${RUNPOD_POD_ID} will self-${POD_GUARD_MODE:-remove} on exit" \
         "(grace ${POD_GUARD_GRACE:-60}s)."
}
