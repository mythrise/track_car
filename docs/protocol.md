# TCP Protocol

The Mac is the server. The Raspberry Pi is the client.

All payloads are framed as:

```text
uint32 big-endian payload_length
payload bytes
```

## Session Flow

```text
Pi -> Mac: hello JSON
loop:
  Pi -> Mac: JPEG frame
  Mac -> Pi: command JSON
```

## Hello JSON

```json
{
  "type": "hello",
  "protocol": 1,
  "instruction": "follow the person",
  "width": 320,
  "height": 240
}
```

## Command JSON

```json
{
  "type": "command",
  "seq": 1,
  "motors": [1500, 1500, 1500, 1500],
  "pan": 1500,
  "tilt": 1500,
  "fps": 10.0,
  "confidence": 1.0,
  "mode": 0,
  "stop": true
}
```

## Motor Field

`motors` maps to four vendor channels:

```text
#006PxxxxT0000!#007PxxxxT0000!#008PxxxxT0000!#009PxxxxT0000!
```

The runtime follows the actual car motor mapping:

```text
left wheels:  PWM = 1500 + speed
right wheels: PWM = 1500 - speed
```

For `speed=400`, the keyboard primitives produce:

```text
forward:      [1900, 1100, 1900, 1100]
backward:     [1100, 1900, 1100, 1900]
turn_left:    [1100, 1100, 1100, 1100]
turn_right:   [1900, 1900, 1900, 1900]
strafe_left:  [1100, 1100, 1900, 1900]
strafe_right: [1900, 1900, 1100, 1100]
```

Stop command:

```text
#006P1500T1000!#007P1500T1000!#008P1500T1000!#009P1500T1000!
```

## Safety

The Pi client calls `hardware.stop()` when:

- command JSON has `"stop": true`
- Ctrl+C is pressed
- the socket disconnects
- the socket times out
