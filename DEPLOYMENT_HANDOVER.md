# SA-Hstar (F2 / Harness) 推理部署交接文档

**生成时间：** 2026-07-29  
**目标：** 在新电脑上部署 SA-Hstar 推理，用于实车控制  

---

## 一、需要传输的文件

### 1. 代码仓库（Git Clone）

```bash
git clone https://github.com/mythrise/track_car.git
cd track_car
git checkout agent-team/overhaul
```

最新代码已在 2026-07-29 推送到 `agent-team/overhaul` 分支（commit `6ab3cc0`）。

### 2. 推理权重（zip 文件，单独传输）

```
E:\AAAI\track_car_inference_weights_20260729.zip   (5.3 GB)
```

**zip 内容（3个SA-Hstar权重，按性能排序）：**

| 文件名 | 说明 | H1 wMAE ↓ | 推荐用途 |
|--------|------|-----------|---------|
| `Harness_F2_seed0_official_S-SELF_u128.pt` | **官方预注册权重（首选）** | 0.05913 | ✅ 生产推理 |
| `Harness_F2_seed1_S-SELF_u128.pt` | seed1，多seed验证用 | 0.05589 | 可选 |
| `Harness_F2_seed2_S-SELF_u128.pt` | seed2，多seed验证用 | 0.05244 | 可选 |

> **推理只需要 S-SELF arm**，S-CTRL arm 仅用于训练（已不包含在zip中）。

### 3. 基础模型权重（从 HuggingFace/ModelScope 下载）

这些权重不在zip中，需在新电脑上单独下载：

```bash
# 方法一：Python 脚本自动下载
cd track_car
python weights/resolve_weights.py --manifest weights/weights_manifest.example.json

# 方法二：手动下载
# OpenTrackVLA 基础模型
huggingface-cli download omlab/opentrackvla-qwen06b \
  --local-dir third_party/OpenTrackVLA/ckpts_hf/opentrackvla-qwen06b

# Qwen backbone（可选，HF cache 已有则跳过）
huggingface-cli download Qwen/Qwen3-0.6B \
  --local-dir third_party/OpenTrackVLA/ckpts_hf/qwen3-0.6b

# DINOv3 视觉编码器
huggingface-cli download facebook/dinov3-vits16-pretrain-lvd1689m \
  --local-dir weights/modelscope/dinov3-vits16-pretrain-lvd1689m

# SigLIP 视觉编码器
huggingface-cli download google/siglip-so400m-patch14-384 \
  --local-dir third_party/OpenTrackVLA/ckpts_hf/siglip-so400m-patch14-384
```

---

## 二、环境配置

### 2.1 Python 环境（推荐 conda）

```bash
conda create -n pytorch python=3.11
conda activate pytorch

# PyTorch with CUDA 12.4
pip install torch==2.6.0 torchvision --index-url https://download.pytorch.org/whl/cu124

# 项目依赖
cd track_car
pip install -r requirements.txt   # 如果有
# 或手动安装关键包：
pip install transformers accelerate pillow numpy scipy
```

**验证环境：**
```python
import torch
print(torch.cuda.is_available())           # 必须 True
print(torch.cuda.get_device_name(0))       # 打印GPU名称
print(torch.__version__)                   # 2.6.0+cu124
```

### 2.2 权重部署

```bash
cd track_car

# 1. 解压 Harness 权重
unzip track_car_inference_weights_20260729.zip -d experiments/inference_ckpts/

# 目录结构：
# experiments/inference_ckpts/
# ├── Harness_F2_seed0_official_S-SELF_u128.pt  ← 主推理权重
# ├── Harness_F2_seed1_S-SELF_u128.pt
# └── Harness_F2_seed2_S-SELF_u128.pt

# 2. 验证权重完整性（SHA256）
python -c "
import hashlib, pathlib
for name, expected in [
    ('experiments/inference_ckpts/Harness_F2_seed0_official_S-SELF_u128.pt',
     'b03a70eb...'),  # 替换为实际SHA
]:
    sha = hashlib.sha256(pathlib.Path(name).read_bytes()).hexdigest()
    print(name, 'OK' if sha.startswith(expected[:8]) else 'MISMATCH')
"
```

---

## 三、推理启动

### 3.1 F2/SA-Hstar 推理模式（新方法）

SA-Hstar 使用 `f2_experiment` 推理路径（不是老的 mac_server.py PFEM 路径）：

```bash
cd track_car
conda activate pytorch

# 启动推理服务（监听Pi摄像头帧）
python inference_pipeline/mac_server.py \
  --ckpt experiments/inference_ckpts/Harness_F2_seed0_official_S-SELF_u128.pt \
  --base_hf_model_dir third_party/OpenTrackVLA/ckpts_hf/opentrackvla-qwen06b \
  --dinov3_model_path weights/modelscope/dinov3-vits16-pretrain-lvd1689m \
  --port 9999 \
  --device cuda \
  --history 31 \
  --state_mode rolling
```

**关键参数说明：**

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--ckpt` | SA-Hstar S-SELF 权重路径（必须） | None |
| `--base_hf_model_dir` | OpenTrackVLA 基础模型目录（必须） | None |
| `--port` | 监听端口（Pi 客户端需对应） | 9999 |
| `--device` | `cuda` 或 `cpu`（推荐 cuda） | auto |
| `--history` | 帧历史长度 | 31 |
| `--state_mode` | `rolling`（保持状态）或 `stateless` | stateless |
| `--control_dt` | 控制周期（秒） | 0.1 |
| `--motor_scale` | 电机幅度缩放 | 400.0 |

### 3.2 协议测试（不加载模型）

```bash
python inference_pipeline/mac_server.py \
  --mock_control \
  --mock_action forward \
  --port 9999
```

### 3.3 Pi 端客户端（不变）

```bash
# 在 Raspberry Pi 上
python car_runtime/pi_client.py \
  --server_host <mac_ip> \
  --server_port 9999
```

---

## 四、权重说明（用哪个）

### 推荐：`Harness_F2_seed0_official_S-SELF_u128.pt`

这是**官方预注册权重**，在 AAAI 论文中作为 SA-Hstar 的主要结果使用：

- H1 wMAE = **0.05913**（比 B0 基线低 61.5%）
- 在2,848行公开验证集上验证
- 架构：L1+D2+AP2+F2，128次优化器更新

### 替代：seed1 (H1=0.05589) 或 seed2 (H1=0.05244)

seed2 的验证集指标最低，但3个seed方向一致。如果 seed0 在实车上表现不佳，可以尝试其他seed。

### 不要用于推理的权重

- `NoAux_seed*.pt` — 这是消融实验权重（无辅助头），性能差于官方版
- `B0_seed*.pt` / `B1_seed*.pt` — 基线，不是SA-Hstar
- `checkpoint_update0_*.pt` — 训练前初始权重，没有意义

---

## 五、目录结构（部署后）

```
track_car/
├── CLAUDE.md                          ← 实验日志（本次会话新增）
├── DEPLOYMENT_HANDOVER.md             ← 本文件
├── f2_experiment/                     ← SA-Hstar 核心代码
│   ├── assembly.py
│   ├── assembly_model.py
│   ├── assembly_data.py
│   ├── validation_diagnostics.py
│   └── ...
├── inference_pipeline/
│   └── mac_server.py                  ← 推理服务入口
├── car_runtime/
│   └── pi_client.py                   ← Pi 端客户端
├── scripts/                           ← 训练/评测脚本（本次会话新增）
│   ├── train_f2_seeded.py
│   ├── eval_harness_multiseed.py
│   ├── train_noaux_seed0.py
│   └── ...
├── third_party/OpenTrackVLA/
│   └── ckpts_hf/                      ← 基础模型（需下载）
├── weights/
│   └── modelscope/                    ← DINOv3（需下载）
└── experiments/
    └── inference_ckpts/               ← 解压zip后的推理权重
        └── Harness_F2_seed0_official_S-SELF_u128.pt
```

---

## 六、排障清单

**问题：CUDA not available**
```bash
python -c "import torch; print(torch.cuda.is_available())"
# 确认安装了 +cu124 版本的 torch
pip install torch==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124
```

**问题：模型加载报 key mismatch**
- 确认使用的是 `S-SELF` arm 权重（不是 `S-CTRL`）
- 权重文件名应包含 `S-SELF`
- 检查 `checkpoint_sha256` 与本文档一致

**问题：Pi 无法连接**
```bash
# 检查端口是否开放
python inference_pipeline/mac_server.py --mock_control --port 9999
# 检查防火墙：允许9999端口TCP入站
```

**问题：推理延迟高**
- 确认 `--device cuda`（不是 cpu）
- GPU内存需 ≥ 6GB（SA-Hstar 约需 5GB）

---

## 七、实验数据（如需复现）

所有实验结果在：
```
E:\AAAI\track_car\experiments\windows_cuda_f2\public_val_memory_reasoning_v1\multiseed_eval\
├── ALL_EXPERIMENT_RESULTS_ASCII.json  ← 所有实验数据（已传给论文写作）
├── MULTISEED_HANDOFF.md
└── 交接文档_多seed实验.md
```

**核心指标（2848行公开验证集，H1 source-macro wMAE ↓）：**

| 方法 | mean H1 | std H1 |
|------|---------|--------|
| B0 (baseline) | 0.15378 | 0.000 |
| NoAux (消融) | 0.06773 | 0.004 |
| **SA-Hstar (Harness)** | **0.05582** | **0.003** |

---

*本文档由 Claude Opus 5 生成，2026-07-29*
