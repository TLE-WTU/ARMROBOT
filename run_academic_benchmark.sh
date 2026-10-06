#!/usr/bin/env bash
# ==============================================================================
# Script chạy Bộ Đánh Giá Thực Nghiệm Chuẩn Quốc Tế Cho Tay Gắp Robot
# (Peer-Reviewed Academic Benchmark: Geometric vs Deep Learning AnyGrasp)
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_EXEC="/home/tienle/miniconda3/envs/robot_env/bin/python"

if [ ! -f "$PYTHON_EXEC" ]; then
    echo "❌ Không tìm thấy Python môi trường robot_env tại $PYTHON_EXEC"
    exit 1
fi

echo "========================================================================"
echo "🚀 KHỞI ĐỘNG BỘ ĐÁNH GIÁ CHUẨN QUỐC TẾ (ACADEMIC GRASP BENCHMARK)"
echo "   Môi trường: Conda robot_env (Python 3.10 + PyBullet + AnyGrasp + PyTorch)"
echo "========================================================================"

"$PYTHON_EXEC" "$SCRIPT_DIR/benchmark/run_academic_benchmark.py" "$@"
