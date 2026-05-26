# Data Pipeline

This pipeline records what the car sees and what command was applied, then
converts each episode into OpenTrackVLA-style training samples.

## Raw Episode Layout

`collect_data.py` writes:

```text
data/collected/<episode>/
  frame_000000.jpg
  meta_000000.json
  frame_000001.jpg
  meta_000001.json
  ...
  episode.json
```

Each `meta_*.json` contains:

```json
{
  "frame": "frame_000000.jpg",
  "timestamp": 0.0,
  "frame_idx": 0,
  "instruction": "follow the person",
  "episode": "ep001",
  "teleop": "keyboard",
  "command": "forward",
  "motors": [1700, 1300, 1700, 1300],
  "action": [1.0, 0.0, 0.0]
}
```

## Collect

On Raspberry Pi:

```bash
python3 data_pipeline/collect_data.py \
  --episode_name ep001 \
  --instruction "follow the person in red shirt" \
  --teleop keyboard \
  --dry_run \
  --speed 200 \
  --fps 10
```

Use `--dry_run` to record without moving hardware.

If startup pauses after UART opens, it is usually waiting for the camera. Test
camera latency directly:

```bash
python3 car_runtime/camera_check.py
python3 car_runtime/camera_check.py --camera_backend picamera2
python3 car_runtime/camera_check.py --camera_backend v4l2
```

For Raspberry Pi CSI cameras, prefer `picamera2`. Install it if needed:

```bash
sudo apt update
sudo apt install -y python3-picamera2
```

Then use the working backend in collection:

```bash
python3 data_pipeline/collect_data.py \
  --episode_name ep001 \
  --teleop keyboard \
  --speed 160 \
  --camera_backend picamera2
```

If low-speed `w/a/s/d` commands do not overcome static friction, add a short
startup kick:

```bash
python3 data_pipeline/collect_data.py \
  --episode_name ep001 \
  --instruction "follow the person in red shirt" \
  --teleop keyboard \
  --speed 160 \
  --kick_speed 350 \
  --kick_duration 0.06 \
  --kick_repeat 0.75 \
  --fps 10
```

This sends the stronger command for only `0.06` seconds, then returns to the
steady `--speed`. Start with the car lifted.

`collect_data.py` cleans stale vendor `mjpg` and `z_main` processes before
opening the camera. Preview cleanup targets:

```bash
python3 data_pipeline/collect_data.py \
  --episode_name cleanup_preview \
  --teleop none \
  --cleanup_dry_run
```

Only remove `--dry_run` after `python3-serial` is installed and
`car_runtime/hardware_check.py --open_uart` can open the UART port.

## Troubleshooting

If you see an error like:

```text
AttributeError: 'NoneType' object has no attribute 'setup_uart'
```

you tried real motor control without the vendor UART module. For data-pipeline
testing, rerun the command with `--dry_run`. In the current runtime, this has
been replaced by direct pyserial UART output, so update the repository and
install serial support:

```bash
sudo apt update
sudo apt install -y python3-serial
python3 car_runtime/hardware_check.py --open_uart
```

## Convert

On Mac:

```bash
python data_pipeline/build_training_data.py \
  --input data/collected \
  --output data/car_train.jsonl
```

The output contains:

```text
current
images
instruction
waypoints
actions
motors
command
polar_theta_idx
polar_dist_idx
polar_invalid
```

Current waypoints are integrated from logged normalized actions. They are not
meter-accurate odometry; treat them as imitation-control targets until the car
has calibrated odometry.

## Known Limitations

- The current target detector is a simple Haar face detector.
- For formal experiments, replace it with a person detector/tracker or manual
  annotation.
- There is no motor feedback or command acknowledgement.
