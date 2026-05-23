# Weights

Do not commit model weights to this repository.

This directory tracks what weights the project expects and where they should
live on the machine.

## Required / Expected Weights

| Name | Purpose | Local path example |
| --- | --- | --- |
| OpenTrackVLA Qwen0.6B | Base model / baseline / initialization | `/Users/mythrise/科研实习/OpenTrackVLA/ckpts_hf/opentrackvla-qwen06b` |
| PFEM checkpoint | Trained harness weights | `/Users/mythrise/科研实习/OpenTrackVLA/ckpts_pfem/pfem_epoch3.pt` |
| SigLIP/DINOv2 | Vision tokenization | Managed by OpenTrackVLA / Hugging Face cache |

## Git Ignore Policy

The repository ignores:

```text
*.pt
*.pth
*.safetensors
ckpts*/
data/
sim_data/
world_rollouts/
videos/images
```

Use `weights_manifest.example.json` as the template for local configuration.

Create and validate a local manifest:

```bash
cp weights/weights_manifest.example.json weights/weights_manifest.local.json
# edit weights/weights_manifest.local.json
python weights/resolve_weights.py --manifest weights/weights_manifest.local.json
```

To inspect the example manifest without failing on placeholder paths:

```bash
python weights/resolve_weights.py --manifest weights/weights_manifest.example.json --allow_missing
```
