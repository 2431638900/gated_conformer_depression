# =============================================================================
# 一键训练 + 测试脚本 (Windows): HuBERT 完全冻结实验
#   - 仅训练：加权层聚合 (layer_weights) + DepressionFeatureExtractor
#             + GatedConformer + StatPooling + Classifier
#   - HuBERT 12 个 transformer 层全部 requires_grad=False
#   - checkpoint 保存为 best_model.pth, 不覆盖原 best_model.pth (5 层版)
# =============================================================================
$ErrorActionPreference = 'Stop'

# ---------- 1) 进入项目根目录 ----------
$ScriptDir   = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = (Resolve-Path (Join-Path $ScriptDir '..')).Path
Set-Location $ProjectRoot
Write-Host "[INFO] 项目根: $ProjectRoot"

# ---------- 2) 激活 conda 环境 ----------
$CondaEnv = if ($env:CONDA_ENV) { $env:CONDA_ENV } else { 'depression_detection' }
if ($env:CONDA_DEFAULT_ENV -ne $CondaEnv) {
    # 寻找 conda 安装位置
    $CondaExe = (Get-Command conda -ErrorAction SilentlyContinue).Source
    if (-not $CondaExe) {
        Write-Error "未找到 conda, 请先安装/把 conda 加入 PATH"
        exit 1
    }
    # 加载 conda hook
    & conda shell.powershell hook | Out-String | Invoke-Expression
    conda activate $CondaEnv
}
Write-Host "[INFO] Python: $(python -c 'import sys; print(sys.executable)')"
Write-Host "[INFO] CUDA: $(python -c 'import torch; print(torch.cuda.is_available(), torch.cuda.device_count())')"

# ---------- 3) 训练 (HuBERT 全冻结) ----------
Write-Host '============================================================'
Write-Host ' 启动训练: --finetune-layers 0  (HuBERT 全冻结)'
Write-Host '============================================================'
python scripts/02_train_model.py --finetune-layers 0
if ($LASTEXITCODE -ne 0) { Write-Error '训练失败'; exit $LASTEXITCODE }

# ---------- 4) 测试 (使用刚训出的 ft0 checkpoint) ----------
Write-Host '============================================================'
Write-Host ' 启动测试: best_model.pth'
Write-Host '============================================================'
python scripts/03_test_model.py --checkpoint best_model.pth --log-suffix _ft0
if ($LASTEXITCODE -ne 0) { Write-Error '测试失败'; exit $LASTEXITCODE }

Write-Host '[OK] 全部完成。'
Write-Host '[OK] checkpoint: checkpoints/best_model.pth'
Write-Host '[OK] 测试日志:   logs/test_*_ft0.log'
Write-Host '[OK] 测试结果:   logs/test_results_*_ft0.json'
