# Deployment Handover — SA-Hstar (F2/Harness) Inference

**Date:** 2026-07-29  
**Model:** SA-Hstar (F2, L1+D2+AP2+F2 architecture lock)  
**Best checkpoint:** Harness_F2_seed0_official (H1 wMAE = 0.05913, 61.5% better than B0)

---

## 1. 文件清单

### 需要从源机器拷贝的文件

**权重包（已打包）：**
```
E:\AAAI\track_car_inference_weights_20260729.zip   (5.3 GB)
内含：
  Harness_F2_seed0_official_S-SELF_u128.pt   ← 主推理权重（预注册seed0）
  Harness_F2_seed1_S-SELF_u128.pt            ← 备用seed1
  Harness_F2_seed2_S-SELF_u128.pt            ← 备用seed2
```

**代码（git clone）：**
```
https://github.com/mythrise/track_car.git
分支: agent-team/overhaul
最新commit: 6ab3cc0 (Add multi-seed training/eval scripts, NoAux ablation, CLAUDE.md)
```

**需要单独下载的基础模型（HuggingFace / ModelScope）：**
```
omlab/opentrackvla-qwen06b          → third_party/OpenTrackVLA/ckpts_hf/opentrackvla-qwen06b/
Qwen/Qwen3-0.6B                     → third_party/OpenTrackVLA/ckpts_hf/qwen3-0.6b/
google/siglip-so400m-patch14-384    → third_party/OpenTrackVLA/ckpts_hf/siglip-so400m-patch14-384/
facebook/dinov3-vits16-pretrain-lvd1689m → weights/modelscope/dinov3-vits16-pretrain-lvd1689m/
```

---

## 2. 环境搭建

### Python 环境

```bash
# 推荐使用 conda
conda create -n trackvla python=3.11
conda activate trackvla

# 安装 PyTorch（需要CUDA 12.x）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 安装项目依赖
cd track_car
pip install -r requirements.txt   # 如果存在
# 或手动安装关键依赖：
pip install transformers accelerate pillow numpy
```

### 克隆代码

```bash
git clone https://github.com/mythrise/track_car.git
cd track_car
git checkout agent-team/overhaul
```

### 放置权重

```bash
# 解压权重包到项目根目录下的临时目录
mkdir -p ckpts_inference
cd ckpts_inference
unzip /path/to/track_car_inference_weights_20260729.zip

# 验证文件
ls -lh
# 应看到：
# Harness_F2_seed0_official_S-SELF_u128.pt  (~1.6GB)
# Harness_F2_seed1_S-SELF_u128.pt           (~1.6GB)
# Harness_F2_seed2_S-SELF_u128.pt           (~1.6GB)
```

---

## 3. 权重文件说明

| 文件名 | 用途 | H1 wMAE | 说明 |
|--------|------|---------|------|
| `Harness_F2_seed0_official_S-SELF_u128.pt` | **主推理权重** | 0.05913 | 预注册seed0，论文主结果 |
| `Harness_F2_seed1_S-SELF_u128.pt` | 备用 | 0.05589 | 多seed验证用 |
| `Harness_F2_seed2_S-SELF_u128.pt` | 备用 | 0.05244 | 多seed验证用（最优） |

**推理时使用 seed0（预注册）或 seed2（数值最优）均可。**

---

## 4. 权重加载方式

### 用 eval_harness_multiseed.py 验证权重

```bash
cd track_car
python scripts/eval_harness_multiseed.py \
  --checkpoint ckpts_inference/Harness_F2_seed0_official_S-SELF_u128.pt \
  --seed 0 \
  --method-name Harness \
  --output-dir /tmp/eval_out
```

预期输出：
```
[eval_harness] RESULT seed=0
  H1  source-macro wMAE : 0.05912906
  All8 source-macro wMAE: 0.11994564
```

### 用 mac_server.py 启动推理服务

```bash
# 基本启动（需要先配置权重路径）
python inference_pipeline/mac_server.py \
  --ckpt ckpts_inference/Harness_F2_seed0_official_S-SELF_u128.pt \
  --base_hf_model_dir third_party/OpenTrackVLA/ckpts_hf/opentrackvla-qwen06b \
  --dinov3_model_path weights/modelscope/dinov3-vits16-pretrain-lvd1689m \
  --port 9999 \
  --history 31 \
  --device cuda:0
```

### 用 f2_experiment/validation_diagnostics.py 完整评测

```bash
python -m f2_experiment.validation_diagnostics \
  --checkpoint ckpts_inference/Harness_F2_seed0_official_S-SELF_u128.pt \
  --arm S-SELF \
  --snapshot 128
```

---

## 5. 权重格式说明

SA-Hstar checkpoint（S-SELF 推理臂）内部结构：

```python
import torch
ckpt = torch.load("Harness_F2_seed0_official_S-SELF_u128.pt", map_location="cpu")
print(ckpt.keys())
# dict_keys(['model', 'optimizer', 'arm', 'u_pre', 'assembly_receipt_sha256', ...])

# 主要字段：
# ckpt['arm']   == 'S-SELF'
# ckpt['u_pre'] == 128
# ckpt['model'] == state_dict with keys like 'adapter.*', 'model.*'
```

使用 `build_eval_row_predictor_from_checkpoint` 加载用于推理：

```python
from f2_experiment.assembly_model import build_eval_row_predictor_from_checkpoint
import json, torch

receipt = json.load(open("experiments/windows_cuda_f2/assembly_receipt_cuda_final_v1.json"))
payload = torch.load("ckpts_inference/Harness_F2_seed0_official_S-SELF_u128.pt", map_location="cpu", weights_only=True)
predictor = build_eval_row_predictor_from_checkpoint(
    project_root=".",
    receipt_document=receipt,
    arm="S-SELF",
    payload=payload,
    device=torch.device("cuda:0"),
)
```

---

## 6. 数据完整性校验

使用以下SHA256验证权重未损坏：

| 文件 | SHA256 前24位 |
|------|-------------|
| smoke_cuda_v1/checkpoint_update128_S-SELF.pt（seed0原始） | 见 assembly_receipt |
| Harness_F2_seed1 | 9f9fdda71d57590a294cc7a5 |
| Harness_F2_seed2 | 5ca37021236625b99cbd16bf |

```bash
# Linux/Mac 验证
sha256sum Harness_F2_seed0_official_S-SELF_u128.pt

# Windows PowerShell 验证
Get-FileHash Harness_F2_seed0_official_S-SELF_u128.pt -Algorithm SHA256
```

---

## 7. 关键配置文件路径

部署后需确认以下路径存在：

```
track_car/
├── experiments/windows_cuda_f2/
│   └── assembly_receipt_cuda_final_v1.json   ← 必须存在，用于模型构建
├── third_party/OpenTrackVLA/                 ← 基础模型代码
├── f2_experiment/                            ← SA-Hstar 推理代码
│   ├── assembly_model.py
│   ├── validation_diagnostics.py
│   └── ...
└── inference_pipeline/
    └── mac_server.py                         ← 在线推理入口
```

---

## 8. 性能参考

在冻结2848行公开验证集上（H1 source-macro wMAE ↓）：

| 模型 | H1 wMAE | 相对B0改进 |
|------|---------|-----------|
| B0 (baseline) | 0.15378 | — |
| **SA-Hstar seed0** | **0.05913** | **−61.5%** |
| SA-Hstar seed1 | 0.05589 | −63.7% |
| SA-Hstar seed2 | 0.05244 | −65.9% |
| SA-Hstar mean±std | 0.0558±0.0033 | −63.7%±2.1% |

消融结果（同等架构，L_aux=0）：

| 变体 | H1 wMAE | 说明 |
|------|---------|------|
| NoAux (arch only) | 0.0677±0.0041 | 架构贡献87.8% |
| Full Harness | 0.0558±0.0033 | 辅助头额外贡献12.2% |

---

## 9. 常见问题

**Q: CUDA out of memory？**  
A: SA-Hstar推理需约1.4GB显存（单臂），确保GPU空闲内存≥2GB。

**Q: `F2AssemblyContractError: smoke package must be 'SA-Hstar-v1'`？**  
A: 使用了错误的receipt文件。必须用 `assembly_receipt_cuda_final_v1.json`，不能用其他receipt。

**Q: `TokenHashLedger failed` 警告？**  
A: 正常，eval脚本会自动从缓存重建token ledger，功能不受影响。

**Q: 推理速度？**  
A: RTX 4060 Laptop: ~8.8 rows/s（含视觉token缓存）；冷启动（加载609M模型）约3-4分钟。

---

## 10. Git 仓库信息

```
远程: https://github.com/mythrise/track_car.git
分支: agent-team/overhaul
最新commit: 6ab3cc0

关键新增文件（本次实验产生）：
  CLAUDE.md                                  实验状态跟踪
  scripts/train_f2_seeded.py                 多seed训练
  scripts/eval_harness_multiseed.py          SA-Hstar评测（支持--method-name参数）
  scripts/run_multiseed_training.py          B0多seed训练
  scripts/train_noaux_seed0.py               NoAux消融训练
  scripts/analyze_action_change_exact.py     精确动作变化切片分析

实验结果文件（不在git，需单独传输）：
  experiments/windows_cuda_f2/public_val_memory_reasoning_v1/multiseed_eval/
    ALL_EXPERIMENT_RESULTS_ASCII.json        所有实验数据汇总
```
