# Quick Start (CLI)

## Environment
Follow the setup instructions in `./README.md` (create the Python environment, install dependencies, download weights). Run the commands below from the repository root once the required checkpoints (e.g., `Model/VGGT_Accelerator/checkpoint.pth`) are in place.

## Run CoMe Inference (VGGT + Accelerator)
`Test/inference.py` contains a self-contained ETH3D example (botanical_garden, 16 frames, 378×504). Update the configuration block at the top of the script if you need to change image directory, sequence length, output directory, or accelerator checkpoint. Then execute:

```bash
PYTHONPATH=/workspace python Test/inference.py
```

Outputs are written to `./inference_outputs/<scene_name>/`:
- `inference_result_seq_<S>.pth`: aggregated dict with depth, world points, confidences, CoMe mask, and metadata.
- `depth_conf.pt`: raw `[B, S, H, W]` VGGT confidence tensor.
- `come_mask.pt`: reshaped `[B, S, tok_h, tok_w]` CoMe mask tensor (yellow = keep, black = merge).
- `visualizations/`: populated in the visualization step below.

## Visualize VGGT Confidence and CoMe Masks
Point the visualization script to the scene directory you just generated to produce per-frame PNGs:

```bash
PYTHONPATH=/workspace python Test/visualize.py \
    --result_dir ./inference_outputs/botanical_garden
```

This creates `visualizations/` inside the scene folder containing:
- `confidence_maps/`: per-frame VGGT confidence heatmaps.
- `come_masks/`: per-frame CoMe masks in yellow/black.
- `comparisons/`: side-by-side panels (confidence + mask) for each frame.

Repeat the command with different `--result_dir` values for other scenes. Use standard tools (e.g., `ffmpeg`, ImageMagick) if you want to stitch the PNGs into videos or GIFs.
