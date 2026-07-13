# Agent Team 总体规划方案（v1，Fable 起草）

> 日期：2026-07-13。分支：`agent-team/overhaul`。
> 目标：①修复现有链路缺陷；②将训练目标改造为"EEF 式相对位移/速度增量"，并将前进与转向解耦训练。

## 0. 背景与动机

现状（codex gpt-5.6-sol 全库分析 + Fable 核实）：

- 训练标签 `waypoints[k] = Σ_{j≤k} action_j·dt` 是**累计位移**（`build_training_data.py:101-117`）。
- 数据严重失衡：forward 1525 条 vs 左右转 271 条，无停车/倒车样本。
- 直行时累计位移单调增长，waypoint MSE 被前进通道主导，转向信号被淹没——**这是"track 学不到真正方向改变"的直接原因**。
- PFEM 训练中 base planner 冻结（`harness_wrapper.py:51-53`），实际学习方向控制的容量非常有限。

改造思路：

1. **增量化（EEF-style delta）**：标签从累计量改为 per-step 增量。匀速直行时速度增量恒为 0，转向事件成为唯一非零信号，方向学习信噪比大幅提升。
2. **前进/转向结构解耦**：新增可训练的 Delta 头，拆成 forward 子头（dim 0）与 steer 子头（dims 1,2），分开 loss + 转向样本加权，前进梯度不再稀释转向学习。

## 1. 团队与职责

| 成员 | 模型 | 职责 |
|---|---|---|
| 成员一 | Claude Fable 5（主会话） | 规划、任务下发、验证、综合裁决（权重高） |
| 成员二 | codex `gpt-5.6-sol`，reasoning effort=xhigh | 按规格实现代码（权重高） |
| 成员三 | agy Gemini 3.1 Pro (High) + Claude Opus 4.8 | 代码审查 |
| 成员四（主席团） | Sonnet 4.5 / Opus 4.8 / codex / agy Gemini | 定期评审方案、任务、实现 |

流程：Fable 规划 → 主席团评审方案 → 团队一修复（codex）→ Fable 验证 → 团队二改造（codex）→ Fable 验证 → 团队三双模型审查 → codex 修复回流 → 主席团终审。

## 2. WS1：修复现有问题（团队一）

全部在分支 `agent-team/overhaul` 上进行。按优先级：

### F1（P0）在线/训练 coarse history 对齐
- 现状：训练 coarse=`t-31…t-1`（`build_training_data.py:166-172`）；在线先 append 当前帧再 build（`mac_server.py:286-290`），当前帧被输入两次。
- 修复：`handle_connection` 中先用**不含当前帧**的历史 build tokens（历史为空时用当前帧 coarse 填充,与 dataset 的 first-frame padding 语义一致），推理后再把当前帧 coarse 追加进历史。
- 验收：新增单元测试模拟 3 帧序列，断言第 k 帧推理时 coarse 序列不含第 k 帧特征。

### F2（P0）真实模型模式的安全停车
- 现状：`stop` 仅在 mock 模式为真（`mac_server.py:324`）；Pi 先执行 motors 再看 stop（`pi_client.py:158` 附近）。
- 修复（server）：新增参数 `--stop_confidence`（默认 0.3）、`--search_stop_frames`（默认 10）、`--max_waypoint_abs`（默认 2.0）。满足任一即 `stop=true` 且 motors=中值：置信度低于阈值、`invalid_pred` 为真、连续 N 帧处于 SEARCH 模式、waypoint 含 NaN/超界。CAUTIOUS 模式将 motor_scale 乘 0.6。
- 修复（Pi）：`stop=true` 时**跳过** motors 执行，直接发 stop_command；顺序改为先判 stop。
- 验收：单测覆盖四种触发条件；mock 协议回归不变。

### F3（P0）PFEM 状态一致性
- 现状：训练每 batch `init_state`（`train_pfem.py:129`），在线跨 session 累积（`mac_server.py:259`）。
- 修复：server 新增 `--state_mode {stateless,rolling}`，默认 `stateless`（每帧 init_state，与训练分布一致）；`rolling` 保留现行为供实验。
- 验收：单测断言 stateless 模式下相邻两帧的 prev_state 相互独立。

### F4（P0）训练元数据写入 checkpoint 并在部署端读取
- 修复：`train_pfem.py` 保存 checkpoint 时加入 `meta` 字段：`{n_waypoints, history, dt, fps, label_mode, action_semantics, data_stats}`（dt/fps/label_mode/semantics 从 JSONL 首行 meta 或 CLI 读取）。`mac_server.py` 加载 ckpt 后若含 meta 则覆盖 `--control_dt`/`--history` 默认值，冲突时打印告警并以 ckpt 为准（CLI 显式传参可覆写）。
- 验收：加载带 meta 的 ckpt 时日志输出生效的 dt/history 来源。

### F5（P1）云台防漂移
- 现状：`pan += -theta*3` 无死区无置信度门控（`mac_server.py:179-182`）。
- 修复：`invalid_pred` 或 `C < stop_confidence` 时不更新 pan；|theta|<2° 死区；每帧向 1500 回中 1%（可用 `--pan_recenter 0` 关闭）。
- 验收：单测：连续 invalid 帧 pan 不漂移。

### F6（P1）统一两套 action→waypoint 积分器
- 现状：`build_training_data.py:101` 平面直积分；`model.py:49` 按累计 yaw 旋转积分。
- 修复：`build_training_data.py` 改用与 `model.py::integrate_actions_to_waypoints` 相同的 yaw 旋转积分（抽成共享函数放 `data_pipeline/kinematics.py`，双方 import；model.py 保留薄包装以兼容）。
- 验收：同一 action 序列两个入口输出逐元素一致（单测）。

### F7（P1）JSONL/缓存路径可移植
- 修复：`build_training_data.py` 默认写**相对 repo root** 的路径（保留 `--absolute_paths`）；`JsonTrackingDataset` 的 cache 相对化逻辑对 repo-root 相对路径直接使用，绝对路径回退现行为。
- 验收：新构建的 JSONL 无绝对路径前缀；dataset 能加载并命中 cache。

### F8（P2）杂项
- `train_pfem.py` 设备选择加 CUDA 分支：`cuda > mps > cpu`。
- `--lora` 未实现：直接移除该参数并在 help 中注明未来计划（不留死参数）。
- `compute_losses` 的 `L_track` 使用 `valid_mask` 掩码。
- 训练数据统计脚本 `data_pipeline/dataset_stats.py`：输出 command 分布、Polar 有效率、fps 一致性，供每次构建后检查。

## 3. WS2：训练目标改造（团队二）

### D1 数据层：增量标签（`build_training_data.py`）
新增 `--label_mode {absolute,delta}`，默认 `delta`。delta 模式每个样本新增字段：

```jsonc
{
  "label_mode": "delta",
  "dt": 0.1,                       // 1/episode_fps
  "prev_action": [1.0, 0.0, 0.0],  // t-1 帧实际执行的归一化 action（首帧为 [0,0,0]）
  "delta_pos":  [[dx,dy,dyaw] × 8],   // per-step 身体系位移增量 = action_k · dt（EEF 式相对位移）
  "delta_vel":  [[dvx,dvy,dwz] × 8],  // 速度增量：a_t+k − a_t+k−1，k=0 时相对 prev_action
  "action_semantics": "arc_turn_v2" | "spin_v1",  // 按 episode.json 中 turn_yaw_ratio 是否存在判定
  "command": "forward|turn_left|..."   // 已有，保留供采样器用
}
```
- 保留 `waypoints`/`actions` 字段（向后兼容 + 便于对照实验）。
- `episode.json` 缺少语义版本信息时按采集代码版本推断并在构建日志中醒目提示。
- 旧语义（spin_v1）episode 默认**保留但打标**，训练脚本提供 `--semantics_filter arc_turn_v2` 排除。

### D2 模型层：DeltaPlannerHead（`harness_wrapper.py` + 新文件 `harness/core/delta_planner.py`）
- 新增可训练头，输入与现 planner 相同的 `ctx (B,D)`：
  - `forward_head`: LayerNorm→Linear(D,2D)→GELU→Linear(2D,2D)→GELU→Linear(2D, 8×1)→tanh — 输出前进通道增量
  - `steer_head`: 同结构 → 8×2 — 输出 strafe+yaw 增量
  - 输出拼接为 `(B,8,3)` 的 `delta_vel_pred`；`delta_pos_pred = cumsum 前的 per-step (prev_action+cumsum(delta_vel))·dt`（由共享 kinematics 函数计算，用于位移 loss 与部署）。
- `PFEMHarness.__init__` 加 `label_mode` 参数；delta 模式下 `forward_step` 额外返回 `delta_vel` / `delta_pos`，`waypoints` 由增量积分重建（保持下游接口不变，PFEM 内部 last_action 等逻辑不受影响）。
- absolute 模式行为与现在完全一致（回归保证）。

### D3 损失（`harness_wrapper.py::compute_losses`）
delta 模式下 `L_track` 替换为：
```
L_fwd   = MSE(delta_vel_pred[...,0], gt_delta_vel[...,0])
L_steer = MSE(delta_vel_pred[...,1:], gt_delta_vel[...,1:])
L_pos   = MSE(delta_pos_pred, gt_delta_pos)          // 位移增量一致性
L_track = L_fwd + λ_steer·L_steer + λ_pos·L_pos      // λ_steer 默认 2.0，λ_pos 默认 0.5，CLI 可调
```
其余四项 loss 与 Uncertainty Weighting 结构不变（L_track 仍占 log_sigma[0] 槽位）。

### D4 采样平衡（`train_pfem.py`）
- 按样本 `command` 用 `WeightedRandomSampler`（权重 ∝ 1/√freq，封顶 10×），新增 `--balance_sampling`（默认开）。
- 日志每 epoch 打印各 command 的采样占比。

### D5 部署（`mac_server.py`）
- ckpt meta 含 `label_mode=delta` 时：`action_cmd = clamp(prev_action_state + delta_vel_pred[0], ±max_action_abs)`，其中 `prev_action_state` 为服务器维护的上一帧下发 action（stateless 模式下同样维护，属控制量而非模型状态）；不再做 `waypoint/horizon` 除法。
- absolute 模式走现行路径。debug 字段输出 `label_mode`、`delta_vel[0]`、重建 action。

### D6 验证方案（Fable 执行）
1. 单测全绿（新增 tests/ 目录，pytest）。
2. 用现有 test005/test006 重建 delta JSONL，`dataset_stats.py` 检查字段完整性、delta 数值范围（|delta_vel|≤2、直行段 delta_vel≈0 占比）。
3. 冒烟训练：`train_pfem.py --epochs 1` 跑 ≥20 step，loss 有限且下降趋势；ckpt meta 完整。
4. `mac_server.py --mock_control` 协议回归 + 用假帧走完整 delta 推理路径（无权重时允许随机初始化，仅验证张量流）。

## 4. 里程碑与验收

| 阶段 | 产出 | 验收人 |
|---|---|---|
| M1 方案评审 | 主席团意见 + 修订版方案 | Fable |
| M2 WS1 完成 | F1-F8 实现 + 单测 | Fable 运行测试 |
| M3 WS2 完成 | D1-D5 实现 + 单测 + 冒烟训练 | Fable 运行验证 |
| M4 审查回流 | Gemini/Opus 审查问题清单 → codex 修复 | Fable 复核 |
| M5 终审 | 主席团评测报告 | 用户 |

## 5. 风险

- 真实权重（Qwen/SigLIP/官方 base）不全在本机时，冒烟训练可能只能用随机初始化验证张量流——验收标准相应降级为"数值有限、形状正确"。
- delta 模式对采集 fps 抖动更敏感（dt 不均匀）→ meta 中记录逐帧 timestamp 的 p50/p95 间隔，偏差 >20% 的 episode 构建时警告。
- 旧数据（spin_v1）与新语义混训的风险由打标+过滤开关控制，最终以重采数据为准。
