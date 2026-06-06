# Real Car Inference Safety Checklist

This checklist is for real-car TrackVLA/PFEM deployment. It focuses on the
current failure mode where model inference runs, but the car barely moves.

## 1. Do Not Treat Waypoints As Actions

The model predicts future waypoints:

```text
waypoint[k] = [future_x, future_y, future_yaw]
```

Those values are accumulated displacement over time. They are not motor speed
commands and they are not PWM deltas.

The data pipeline builds these waypoints by integrating logged normalized
actions:

```text
action = [forward, strafe_right, yaw_clockwise]
dt = 1 / fps
waypoint[k].x += forward * dt
waypoint[k].yaw += yaw_clockwise * dt
```

Example with 10 FPS and a held `w` key:

```text
logged action       = [1.0, 0.0, 0.0]
dt                  = 0.1 s
waypoint[0].x       = 0.1
waypoint[1].x       = 0.2
```

If `waypoint[1]` is directly passed to motor conversion, the control code sees
`forward=0.2` instead of `forward=1.0`. With a motor scale of 300, that gives:

```text
0.2 * 300 = 60 PWM delta
```

The real car has already shown that it usually needs roughly 350 to 400 PWM
delta to overcome static friction. A 60 delta can make the motors hum or twitch
without moving the chassis.

Deployment must convert waypoint displacement back into an action-like velocity
before converting it to motor pulses:

```text
action_estimate ~= waypoint[index] / ((index + 1) * control_dt)
motor_delta     ~= action_estimate * motor_scale
```

Recommended initial parameters:

```text
--motor_scale 400
--control_dt 0.1
--control_waypoint_index 1
```

With `waypoint[1].x ~= 0.2`, this gives:

```text
action_estimate = 0.2 / ((1 + 1) * 0.1) = 1.0
motor_delta     = 1.0 * 400 = 400
```

If `python inference_pipeline/mac_server.py --help` does not show these control
parameters, do not run an on-ground real-car test from that server revision. It
means the deployment path may still be using the old direct waypoint-to-motor
logic.

## 2. Windows Inference Server Test Command

Run this on the Windows inference computer, not on the Raspberry Pi:

```powershell
cd C:\Users\29725\Desktop\car\track_car

python inference_pipeline/mac_server.py `
  --port 9999 `
  --timeout 30 `
  --opentrackvla_root third_party/OpenTrackVLA `
  --base_hf_model_dir third_party/OpenTrackVLA/ckpts_hf/opentrackvla-qwen06b `
  --ckpt third_party/OpenTrackVLA/ckpts_pfem/car_official_dinov3/pfem_epoch0.pt `
  --dinov3_model_path weights/modelscope/dinov3-vits16-pretrain-lvd1689m `
  --motor_scale 400 `
  --control_dt 0.1 `
  --control_waypoint_index 1
```

For communication-only testing, use mock mode first:

```powershell
python inference_pipeline/mac_server.py --port 9999 --mock_control --mock_action stop --timeout 30
```

## 3. Raspberry Pi Client Test Command

Run this on the Raspberry Pi:

```bash
cd ~/Desktop/track_car

python3 car_runtime/pi_client.py \
  --server_ip <Windows_IP> \
  --server_port 9999 \
  --instruction "follow the person in red shirt" \
  --camera_backend v4l2 \
  --camera_fourcc MJPG \
  --width 320 \
  --height 240 \
  --timeout 30 \
  --dry_run
```

Recommended real-car kick parameters after dry-run passes:

```bash
--kick_speed 400 \
--kick_duration 0.08 \
--kick_repeat 0.35
```

The kick pulse is only for overcoming static friction. It does not fix wrong
model output or wrong waypoint/action scaling.

## 4. Required Safety Sequence

Always test in this order:

```text
1. dry_run
2. lifted-car
3. on-ground
```

### Stage 1: dry_run

Use `--dry_run` on the Pi. This verifies:

```text
Pi camera -> TCP protocol -> Windows model server -> command JSON -> Pi client
```

The car must not move in this stage.

### Stage 2: lifted-car

Lift the car so all wheels are off the ground. Remove `--dry_run` and add kick
parameters. Verify:

```text
wheels spin in the expected direction
stop command stops all wheels
left/right are not reversed
pan/tilt commands do not collide with the chassis
```

Stop immediately if wheel direction is wrong or if the command oscillates
rapidly.

### Stage 3: on-ground

Only place the car on the ground after Stage 1 and Stage 2 pass. Start with a
clear area and a hand near power. Use short runs first.

## 5. If The Car Still Does Not Move

Separate the problem into threshold failure and model-output failure.

### Threshold failure

The model and scaling may be reasonable, but the final PWM delta is still below
the car's physical starting threshold.

Signs:

```text
lifted-car wheels spin
on-ground car hums, twitches, or moves only after a push
motor deltas are consistent but below roughly 350 to 400
manual move_test works at speed 400
```

Actions:

```text
raise --motor_scale gradually: 400 -> 450 -> 500
keep --control_dt matched to the training FPS
increase kick only slightly: --kick_speed 420, then 450 if needed
verify battery voltage and wheel friction
```

Do not solve threshold failure by using very long kick durations. Long kicks can
hide bad control output and make the car unsafe.

### Model-output failure

The model is producing near-zero, unstable, or wrong-direction waypoints.

Signs:

```text
lifted-car wheels barely move even with --motor_scale 400
predicted waypoint values stay near zero
commands alternate left/right or forward/backward frame to frame
manual move_test works, but inference commands do not
confidence is very low or invalid/polar target stays invalid
```

Actions:

```text
inspect raw predicted waypoint values before motor conversion
compare waypoint[index] / ((index + 1) * control_dt) with expected action scale
run mock_control forward/left/right to verify Pi motor execution separately
collect more real-car episodes with balanced forward/left/right/stop examples
improve target labels before retraining if polar/target labels are mostly invalid
```

Manual motor baseline:

```bash
python3 car_runtime/move_test.py --move forward --speed 400 --duration 0.2 --execute
```

If this baseline cannot move the car on the ground, the issue is hardware,
battery, friction, UART, or motor threshold rather than model inference.

If this baseline moves the car but inference does not, the issue is model output
or the waypoint-to-action-to-motor conversion.
