# PFEM-Harness on OpenTrackVLA

## 完整科研项目运行指南

### 目录结构

```
OpenTrackVLA/
├── model.py                    # 原始 OpenTrackVLA 模型
├── train.py                    # 原始训练脚本
├── trained_agent.py            # 原始 Habitat 推理 agent
│
├── harness/                    # PFEM-Harness 全部模块
│   ├── base_repro/
│   │   ├── polar_cot.py        # Polar-CoT 头 (60θ+30d+invalid)
│   │   └── tim.py              # TIM (4-token 乘性门)
│   ├── core/
│   │   ├── future_module.py    # 层级多 horizon Future (Δ∈{4,8,16})
│   │   ├── verifier.py         # 双头 Verifier (q+δ)
│   │   ├── event_bank.py       # Cognitive Event Bank (6 types)
│   │   └── orchestrator.py     # Orchestrator (soft mode + metadata)
│   ├── schedule/
│   │   └── pseudo_labels/
│   │       └── generate_all.py # 伪标签生成
│   └── harness_wrapper.py      # 总 wrapper (把 base + harness 合一)
│
├── scripts/
│   ├── train_pfem.py           # PFEM 训练入口
│   ├── infer_pfem.py           # PFEM 推理 (单帧/目录)
│   └── car/
│       ├── pi_client.py        # 树莓派 C3 小车端 (采集+执行)
│       ├── mac_server.py       # Mac 推理服务端
│       ├── collect_data.py     # 数据采集脚本
│       └── build_training_data.py  # 采集数据 → 训练格式转换
```

### 全流程

#### Step 1: 数据收集 (在树莓派上)

```bash
# 一边用 APP/遥控器操控小车跟踪人，一边录制
python3 scripts/car/collect_data.py \
    --episode_name ep001 \
    --instruction "follow the person in red shirt" \
    --fps 10
```

#### Step 2: 数据标注 (在 Mac 上)

```bash
# 把采集的帧转成 JSONL 训练格式
python scripts/car/build_training_data.py \
    --input data/collected \
    --output data/car_train.jsonl

# 生成伪标签 (polar-CoT + future + event)
python harness/schedule/pseudo_labels/generate_all.py \
    --input data/car_train.jsonl \
    --output data/car_train_labeled.jsonl
```

#### Step 3: 训练 (在 Mac 或 GPU 服务器上)

```bash
# Stage 1: 冻结 LLM, 训练 Harness
python scripts/train_pfem.py \
    --train_json data/car_train_labeled.jsonl \
    --epochs 4 \
    --batch_size 2 \
    --lr 3e-4

# 输出: ckpts_pfem/pfem_epoch0.pt ... pfem_epoch3.pt
```

#### Step 4: 部署到小车 (Mac 做推理服务)

**Mac 端:**
```bash
python scripts/car/mac_server.py \
    --ckpt ckpts_pfem/pfem_epoch3.pt \
    --port 9999
```

**树莓派端:**
```bash
python3 scripts/car/pi_client.py \
    --server_ip <你的Mac的IP> \
    --server_port 9999 \
    --instruction "follow the person in red shirt"
```

#### Step 5: 推理评估 (单帧/目录)

```bash
python scripts/infer_pfem.py \
    --input data/test_frames/ \
    --ckpt ckpts_pfem/pfem_epoch3.pt \
    --instruction "follow the person"
```

### 30 个创新模块 (在 pfem_repo/ 中已验证)

参见 `pfem_repo/RESULTS.md` 和 `pfem_repo/README.md`。优先推荐应用到本项目的模块:

| 优先级 | 模块 | 文件位置 |
|---|---|---|
| ★★★ | FLARE 中层对齐 | pfem_repo/src/v3_iter02/ |
| ★★★ | 数据增强 | pfem_repo/src/v3_iter27/ |
| ★★★ | RoVer 推理重打分 | pfem_repo/src/v3_iter04/ |
| ★★☆ | mode-prior KL | pfem_repo/src/v3_iter26/ |
| ★★☆ | attention pooling | pfem_repo/src/v3_iter25/ |
