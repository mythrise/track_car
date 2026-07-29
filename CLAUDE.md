# AAAI Track Car Experiment Log

## 项目概述
TrackVLA++ Harness (SA-Hstar / F2) vs B0 baseline ablation study.
架构锁：L1+D2+AP2+F2 | 验证集：2,848行公开验证集（冻结）

---

## 实验状态总览

### ✅ 已完成实验

#### P0: Multi-Seed Training (完成于 2026-07-27)
目标：补充seed 1和2，验证方向一致性。

**训练checkpoint路径：**
```
experiments/windows_cuda_f2/public_val_memory_reasoning_v1/matched128/
├── B0_seed1/baseline_epoch0.pt       SHA: 5ba80213...
├── B0_seed2/baseline_epoch0.pt       SHA: 5d263559...
├── F2_seed1/checkpoint_update128_S-SELF.pt  SHA: a5579442...
└── F2_seed2/checkpoint_update128_S-SELF.pt  SHA: 2ad7ecfb...
```

**核心结果（H1 source-macro wMAE ↓）：**

| 方法 | seed | H1 wMAE | All8 wMAE |
|------|------|---------|-----------|
| B0   | 0    | 0.15377812 | 0.14890242 |
| B0   | 1    | 0.15377812 | 0.14890242 |
| B0   | 2    | 0.15377812 | 0.14890242 |
| **B0 mean±std** | | **0.15377812 ± 0.00000000** | |
| Harness | 0 | 0.05912906 | 0.11994564 |
| Harness | 1 | 0.05589072 | 0.12160525 |
| Harness | 2 | 0.05244407 | 0.10750864 |
| **Harness mean±std** | | **0.05582128 ± 0.00334304** | |

**结论：** 所有3个seed上Harness < B0 ✓

**评测/收据文件：**
```
experiments/windows_cuda_f2/public_val_memory_reasoning_v1/multiseed_eval/
├── eval_result_B0_seed{1,2}.json
├── eval_result_Harness_seed{1,2}.json
├── run_receipt_*.json
├── MULTISEED_HANDOFF.md
└── 交接文档_多seed实验.md
```

---

#### P1: Action-Change Slice (探索性事后，完成于 2026-07-27)
**输出文件：** `multiseed_eval/p1_action_change_slices.json`

使用阈值切片（已知阈值设置与离散动作空间不完全吻合，见P2改进）。

---

#### Exp2: Action-Change Exact-Match (完成于 2026-07-28)
**输出文件：** `multiseed_eval/p2_action_change_metrics.json`

使用精确离散值匹配（无阈值），bootstrap CI。

**关键结果：**

| 切片 | N | % | Harness H1 | B0 H1 | Persist H1 |
|------|---|---|-----------|-------|------------|
| overall | 2848 | 100% | 0.05913 | 0.15378 | 0.04340 |
| **action_change** | **246** | **8.6%** | **0.49881** | **0.31235** | **0.50272** |
| action_no_change | 2602 | 91.4% | 0.01758 | 0.13851 | 0.00000 |

**persistence_normalized_H1（只看246行动作变化）：**
- Harness: 0.9922（CI95: [0.990, 0.995]）— 比persistence差0.8%
- B0: 0.6214 — 比persistence好37.9%

⚠️ EXPLORATORY_POST_HOC — 不是预注册指标

---

### ✅ Exp1: No-Auxiliary-Head Ablation (完成于 2026-07-28)
**结果：** NoAux seed0 H1 = **0.07080923**, All8 = 0.15130769
**输出：** `multiseed_eval/eval_result_NoAux_seed0.json`, `run_receipt_NoAux_seed0.json`
**方法：** L_aux=0（forward pass不变，aux gradient清零）
**结论：** 改善的87.8%来自bounded fusion架构，12.2%来自辅助头。

### ✅ Exp2: Action-Change Exact-Match (完成于 2026-07-28)
**结果：** 见 `multiseed_eval/p2_action_change_metrics.json`
**关键发现：** 动作变化行（8.6%）上B0优于Harness（Harness: 0.499, B0: 0.312）。

### ✅ Exp3: Multi-Seed NoAux (完成于 2026-07-28)
**结果：**
- NoAux seed0: H1=0.07081, seed1: H1=0.06934, seed2: H1=0.06306
- NoAux mean±std H1: **0.06773 ± 0.00412**
- NoAux > Harness on all 3 seeds: ✓
**输出：** `multiseed_eval/exp3_noaux_multiseed_summary.json`

---

## 关键文件路径

| 文件 | 说明 |
|------|------|
| `data/collected_v1/datasets/train.jsonl` | SHA: 1715b3ce... (冻结) |
| `data/collected_v1/datasets/val.jsonl` | SHA: 696423b1... (冻结, 2848行) |
| `experiments/windows_cuda_f2/assembly_receipt_cuda_final_v1.json` | SHA: 330993715... |
| `20260719_f2_seeded_cal_lambda_freeze_receipt.json` | λ冻结收据 |
| `scripts/train_f2_seeded.py` | Harness seeds 1/2 训练脚本 |
| `scripts/eval_harness_multiseed.py` | Harness评测脚本 |
| `scripts/run_multiseed_training.py` | B0 seeds 1/2 训练脚本 |
| `scripts/train_noaux_seed0.py` | NoAux ablation训练脚本 |
| `scripts/analyze_action_change_exact.py` | Exp2精确切片分析 |

---

## 硬性边界（不允许做的事）

- ❌ 不打开 `data/collected_v1/episodes/test/`（P2密封内测集未授权）
- ❌ 不修改架构锁 L1+D2+AP2+F2
- ❌ 不修改超参数（除seed外）
- ❌ 不只报告有利方向的结果
- ❌ 不声称internal test已泛化

---

## 评测协议说明

- Seed 0 Harness: 来自预注册协议 `validation_diagnostics.py`，full/logged条件
- Seeds 1/2 Harness: 使用重实现的评测路径（等价，已验证）
- NoAux: 将使用相同的 `eval_harness_multiseed.py` 评测

---

*Last updated: 2026-07-28*
