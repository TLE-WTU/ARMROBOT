#!/usr/bin/env bash
# Benchmark execution wrapper for physical grasp evaluation

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${SCRIPT_DIR}/src/robot_arm:${PYTHONPATH:-}"

PYTHON_EXEC="${PYTHON_EXEC:-python3}"

echo "Starting grasp evaluation benchmark..."
echo "Python: $PYTHON_EXEC"
echo "Working directory: $SCRIPT_DIR"

"$PYTHON_EXEC" "$SCRIPT_DIR/benchmark/run_academic_benchmark.py" "$@"
