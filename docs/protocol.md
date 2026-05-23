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

Stop command:

```text
#255P1500T1000!
```

## Safety

The Pi client calls `hardware.stop()` when:

- command JSON has `"stop": true`
- Ctrl+C is pressed
- the socket disconnects
- the socket times out
