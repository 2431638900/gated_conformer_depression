#!/usr/bin/env bash
# =============================================================================
# 一键训练 + 测试脚本：HuBERT 完全冻结实验
#   - 仅训练：加权层聚合 (layer_weights) + DepressionFeatureExtractor
#             + GatedConformer + StatPooling + Classifier
#   - HuBERT 12 个 transformer 层全部 requires_grad=False，无梯度无优化器 state
#   - checkpoint 保存为 best_model_ft0.pth，不覆盖原 best_model.pth (5 层版)
# =============================================================================
set -e

# ---------- 1) 进入项目根目录 ----------
# 该脚本位于 conformer_depression_v2/scripts/ 下，自动定位到项目根
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "${SCRIPT_DIR}/.." && pwd )"
cd "${PROJECT_ROOT}"
echo "[INFO] 项目根: ${PROJECT_ROOT}"

# ---------- 2) 激活 conda 环境 ----------
# 如果你的环境名不是 depression_detection, 修改下面这行即可
CONDA_ENV="${CONDA_ENV:-depression_detection}"
# 让脚本在子 shell 里也能用 conda activate
if [ -z "${CONDA_DEFAULT_ENV:-}" ] || [ "${CONDA_DEFAULT_ENV}" != "${CONDA_ENV}" ]; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV}"
fi
echo "[INFO] Python: $(which python)"
echo "[INFO] CUDA 可用性: $(python -c 'import torch; print(torch.cuda.is_available(), torch.cuda.device_count())')"

# ---------- 3) 训练 (HuBERT 全冻结) ----------
echo "============================================================"
echo " 启动训练: --finetune-layers 0  (HuBERT 全冻结)"
echo "============================================================"
python scripts/02_train_model.py --finetune-layers 0

# ---------- 4) 测试 (使用刚训出的 ft0 checkpoint) ----------
echo "============================================================"
echo " 启动测试: best_model_ft0.pth"
echo "============================================================"
python scripts/03_test_model.py --checkpoint best_model.pth --log-suffix _ft0

echo "[OK] 全部完成。"
echo "[OK] checkpoint: checkpoints/best_model.pth  (frozen HuBERT)"
echo "[OK] 测试日志:   logs/test_*_ft0.log"
echo "[OK] 测试结果:   logs/test_results_*_ft0.json"
