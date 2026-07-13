# Agent Team 总体规划方案（v2，主席团评审后修订）

> 日期：2026-07-13。分支：`agent-team/overhaul`。
> v1 经主席团四模型评审（Opus 4.8 / Sonnet / codex gpt-5.6-sol / Gemini 3.1 Pro，一致 revise，评分 4-5），
> 本版按权重（Fable、gpt-5.6-sol 高）综合修订。评审原始意见见工作流 wf_b81da021-d62 journal。

## 0. 目标与核心设计（v2 修订）

用户目标：①前进(w)与转向(a/d)解耦训练——前进样本淹没转向学习；②标签改为 EEF 式**相对位移/增量**而非累计绝对量。

### 0.1 为什么 v1 的 delta_vel 主目标被否决
codex 实测 `car_train.jsonl`：14368 个 per-step delta_vel 中 **88.6% 是精确零**（匀速直行与匀速转向的稳态都是 0），
非零只在指令切换的 ~206 个过渡帧。MSE 下"永远输出 0"是平凡最优解，部署闭环 `a_t = a_{t-1} + 0` 即"永远保持上一动作"——
模型从 forward 状态永不主动发起转向，**恰好奖励惰性**。另有 tanh(±1) 装不下 ±2 值域、prev_action 不可观测、开环积分漂移三个连带 blocker。

### 0.2 v2 核心表征：逐步 action 序列（= 逐步相对位移 / dt）
- **主标签** `step_actions[k] = a_{t+k}`，k=0..7，每步 ∈[−1,1]³。它等价于 EEF 式逐步相对位移 `delta_pos[k]=a_{t+k}·dt` 除以 dt——
  仍是"增量/相对量"（每步独立、无累计），但数值稳定、契合 tanh、部署直接 `action_cmd = pred[0]`，**无闭环积分、无漂移**。
- **辅助标签** `delta_vel[k] = a_{t+k} − a_{t+k−1}`（k=0 相对 `prev_action`）：作辅助 loss（权重小）+ 消融开关，
  条件：prev_action 经小 embedding **显式注入 ctx**；头输出 `tanh×2` 缩放；默认关闭（`--aux_delta_vel` 开启）。
- **解耦**：共享小 trunk `LayerNorm(D)→Linear(D,256)→GELU`，再分 `forward_branch: Linear(256,8)`（dim0）与
  `yaw_branch: Linear(256,8)`（dim2），各自 tanh；**strafe(dim1) 冻结输出 0**（无数据）。参数量 ~0.3M（v1 的 12.6M 会背题）。
- **loss 解耦**：`L_fwd = SmoothL1(pred_fwd, gt_fwd)`、`L_yaw = SmoothL1(pred_yaw, gt_yaw)`，
  `L_track = L_fwd + λ_yaw·L_yaw`（λ_yaw 默认 2.0，CLI 可调），占原 log_sigma[0] 槽位。valid_mask 参与掩码。
- **采样平衡**：按**事件分层**而非 command 频率——每样本标 `transition_type ∈ {steady_forward, turn_onset, sustained_turn, turn_exit, other}`
  （由 horizon 内 yaw 通道变化模式判定），采样权重使四类转向事件合计 ≥40%，封顶 10×。`--balance_sampling` 默认开。
- **镜像增强**：左右翻转图像 + yaw/strafe 取反，构建期生成（`--mirror_augment` 默认开，只对含转向的样本），转向样本翻倍。

### 0.3 部署（mac_server）
- ckpt meta `label_mode=step_action` 时：`action_cmd = pred_step_actions[0]`，然后过安全层：
  每轴 rate-limit（`--max_action_rate` /s，按**实际 elapsed 时间**缩放）+ EMA 平滑（`--action_ema`，默认 0.5）。
  这提供了用户想要的"增量式平滑"效果，但以护栏形式实现而非自由积分。
- stop/invalid/低置信度/断连时安全层状态清零。

## 1. 团队与流程（不变）
Fable 规划/验证/裁决（权重高）→ 主席团评审 → codex gpt-5.6-sol xhigh 实现（权重高）→ agy Gemini 3.1 Pro + Opus 4.8 审查 → 主席团终审。

## 2. WS1：修复现有问题（团队一，codex 实现）

### F1（P0）在线/训练 coarse history 对齐 + 启动 warmup
- 在线 coarse 只用 `t-31…t-1`：先 build_tokens（不含当前帧），推理后再 append 当前帧 coarse。
- 历史不足 `--warmup_frames`（默认=history）时输出 stop 命令（motors=中值）不驱车，只积累历史——首帧重复 padding 不属于训练分布。
- 单测：模拟序列断言第 k 帧推理时 coarse 不含第 k 帧特征；warmup 期间输出 stop。

### F2（P0）真实模型模式安全停车（修订：不依赖无监督 mode）
- server 触发条件（任一）：`confidence < --stop_confidence`(默认0.3)、`invalid_pred` 连续 ≥ `--invalid_stop_frames`(默认5)、
  waypoint/action 含 NaN/Inf 或超 `--max_waypoint_abs`(默认2.0)。触发时 `stop=true` 且 motors=中值，并清零安全层状态。
  **不用** orchestrator 的 SEARCH/CAUTIOUS mode（无监督 argmax，不可靠）；mode 仅记入 debug。
- Pi 端：先判 `stop`，为 true 时跳过 motors 直接 stop_command。
- 单测覆盖三种触发 + mock 协议回归。

### F3（P0）PFEM 状态一致性
- `--state_mode {stateless,rolling}` 默认 stateless（每帧 init_state，匹配训练）。rolling 保留但启动时打印"实验性，与训练分布不一致"警告。
- 表述修正（codex）：这是 conditioned-stateless policy——PFEM latent 无状态，控制安全层（rate-limit/EMA/prev cmd）是控制器状态，两者不矛盾。

### F4（P0）checkpoint 元数据（最终：缺失即 fail-closed）
- `train_pfem.py` checkpoint 写入 `meta = {schema_version:1, n_waypoints, history, dt, fps, label_mode, action_semantics, delta_scale, data_manifest_hash, data_jsonl_sha256, sample_count, train_args}`。
- `mac_server.py` 读 meta 设置 dt/history/label_mode；非 mock 模式下 checkpoint 缺 meta，或 meta 缺 schema_version/label_mode/history/dt 任一字段，均拒绝启动，不再默认 absolute。显式 CLI 冲突仍打印强警告并以 CLI 为准。

### F5（P1）云台防漂移
- invalid_pred 或 C<stop_confidence 时不更新 pan；`--pan_deadzone_deg` 默认 4°，可覆盖 Polar 60-bin 解码的最小非零中心角 ±3°；回中速率按**秒**计（`--pan_recenter_per_s`，默认 30 PWM/s，0 关闭）。

### F6（P1）共享运动学 + 修 off-by-one
- 新建 `data_pipeline/kinematics.py`：`integrate_actions(actions, dt) -> waypoints`，语义 `waypoint[k] = compose(action[0..k])`（8 个 action 全用），
  按累计 yaw 旋转到局部起始系（修正 builder 的平面积分在转向轨迹上物理错误的问题，也修 model.py 循环从 t=1 起、最后一个 action 丢失的 off-by-one）。
- builder 与 model.py 都改用它（model.py 留薄包装兼容旧签名）。
- 单测：纯 yaw、yaw→forward、forward→yaw、镜像 round-trip。

### F7（P1）路径可移植 + sidecar manifest
- JSONL 默认写 repo-root 相对路径（保留 --absolute_paths）。
- 数据集元数据写 **sidecar** `<output>.manifest.json`（含 path_root, fps, dt, action_semantics, label_mode, schema_version、JSONL SHA-256、行数与统计信息）——
  **不写 JSONL 首行**（dataset 把每行当样本，会崩）。dataset 加载时若发现同名 manifest 则读取并校验。

### F8（P0 升级）数据完整性 + 杂项
- builder 遇到空/坏 meta、空 episode.json、时间戳间隔 p95/p50>1.2、未知语义时：**默认报错退出**（--lenient 降级为警告），
  输出每 episode 的完整性报告（test006 已知有空 episode.json 和 127 个空 meta）。
- builder 写出后用 `training_sample.schema.json` 校验首样本与随机 3 个样本；train_pfem 强制核对 JSONL SHA-256/行数，step_action 数据缺 step_actions/prev_action/delta_vel 任一字段即拒训，不提供 lenient 降级。
- `data_pipeline/dataset_stats.py`：command/transition_type 分布、Polar 有效率、fps 一致性、delta 值域统计。
- train_pfem 设备选择 cuda>mps>cpu；移除 --lora 死参数；compute_losses 用 valid_mask；requirements.txt 加 pytest（并给关键包加下限版本）。

### F9（P0 新增）伪标签质量：Haar 正脸 → OmDet-Turbo 行人检测
- `build_training_data.py` 的 `estimate_target_from_frame` 改为优先用 OmDet-Turbo（复用 model.py:388 `_get_bbox_detector` 的加载方式，
  提成共享模块 `data_pipeline/target_detector.py`），检测 "person"，取最高分框；模型不可用时回退 Haar 并醒目警告。
- 距离仍为启发式，并按检测来源区分：OmDet 使用完整行人框高度，Haar 回退使用人脸框纵向位置；manifest 标注 `distance_source: source_aware_heuristic` 与分来源规则。
- 目标：Polar 有效率从 25.6% 显著提升；构建报告输出前后对比。

### F10（P0 新增）checkpoint fail-closed
- 非 mock 模式下：PFEM ckpt 不存在、meta 缺失/不完整、或控制关键模块（delta/step 头、context_proj、cot、proj）的 key 缺失 → **拒绝启动**并列出原因；
  `--allow_random_init` 仅在配合 `--shadow_mode`（只打印命令不发 motors，见 F11）时允许。
- base.llm.* 缺失仍容忍（Qwen 独立加载）。

### F11（P1 新增）shadow mode
- `--shadow_mode`：完整推理但对 Pi 只发 stop 命令，把本应发送的 motors/action 写入 debug 与日志。真车前验证用。

## 3. WS2：训练目标改造（团队二，codex 实现）

### D1 数据层（build_training_data.py）
- `--label_mode {step_action,absolute}` 默认 step_action。新增字段：
  `step_actions`(8×3)、`delta_pos`(8×3, =action·dt)、`delta_vel`(8×3, k=0 相对 prev_action)、`prev_action`(3)、
  `transition_type`(str)、`mirrored`(bool)。保留 `waypoints/actions` 兼容。
- `action_semantics`: episode.json 含 turn_yaw_ratio → `arc_turn_v2`，否则 `spin_v1`；写入每样本与 manifest。
  **现存全部转向数据是 spin_v1**——不过滤（否则训练集空），打标并在报告中醒目提示"最终收益依赖重采 arc 语义数据"。
- 镜像增强见 0.2；`--val_episodes test006` 类过滤参数支持按 episode 输出独立 train/val JSONL。

### D2 模型层（harness/core/step_planner.py + harness_wrapper.py）
- `StepActionHead`：共享 trunk D→256 + forward/yaw 分支（各 →8, tanh），strafe 输出常零。
- `PFEMHarness(label_mode=...)`：step_action 模式下 forward_step 返回 `step_actions` 并由 kinematics 重建 `waypoints`（保持下游接口）；
  **base.planner 与 Verifier delta 残差在 step_action 模式下停用**（不叠加，避免打架——v1 欠规格点）；absolute 模式行为不变（回归保证）。
- `--aux_delta_vel` 开启时：prev_action 经 `Linear(3,64)` embedding 拼入 ctx（ctx_dim 相应 +64），
  aux 头输出 `tanh×2`，`L_dvel` 权重 0.2 加入 L_track；训练时 prev_action 来自 JSONL 字段并加 σ=0.05 高斯噪声（scheduled-sampling 式鲁棒化）。

### D3 损失（见 0.2）+ D4 采样（见 0.2）
### D5 部署（见 0.3）
### D6 评测（scripts/eval_offline.py，新增，P0）
- 输入 val JSONL + ckpt，输出：per-axis SmoothL1、**turn-sign accuracy**（|yaw_gt|>0.2 帧上 sign(pred)==sign(gt) 比例）、
  **transition F1**（yaw 通道变化事件的检出）、饱和率（|pred|>0.95 占比）、每 transition_type 分组指标。
- 基线对照（M3 验收必做）：absolute(现状) vs absolute+平衡采样 vs step_action vs step_action+aux_delta_vel 四组，
  同一 held-out（test006）指标对比表。**离线指标不胜出不进真车**；真车前必须过 shadow mode。

## 4. 验证与里程碑
- M2（WS1）：pytest 全绿；mock 协议回归；builder 在 test005/test006 上的完整性报告。
- M3（WS2）：重建数据（含镜像）通过 stats 检查；冒烟训练 ≥20 step loss 有限下降；ckpt meta 完整；eval_offline 四组对照表产出。
- M4：Gemini 3.1 Pro + Opus 4.8 审查 diff → codex 修复 → Fable 复核。
- M5：主席团终审 + 用户汇报。

## 5. 风险与前置依赖（升级为显式声明）
1. **数据是根本瓶颈**：2 个 episode、~100-206 个独立转向机动、全部旧 spin 语义。本改造把管道与表征做对，
   但**收益封顶于数据**。强烈建议 WS0：用新 arc 语义重采 10-20 个 episode（含转弯进入/持续/退出、目标丢失、停车、重捕获）——需要用户实车操作。
2. OmDet-Turbo 权重若本机没有（HF_HUB_OFFLINE=1），F9 自动回退 Haar 并警告，标签质量目标顺延。
3. 冒烟训练若缺 Qwen/SigLIP 权重则降级为张量流验证（--allow_random_init + 不进真车）。
