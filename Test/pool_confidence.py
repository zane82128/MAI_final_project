"""
Average-pool VGGT confidence maps on the patch grid.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F


def pool_confidence(depth_conf: torch.Tensor, patch_size: int) -> torch.Tensor:
    """
    Average-pool the confidence tensor over non-overlapping patches.

    Args:
        depth_conf: [B, S, H, W]
        patch_size: patch edge in pixels (e.g., 14 for VGGT)

    Returns:
        Tensor shaped [B, S, H//patch, W//patch]
    """
    if depth_conf.ndim != 4:
        raise ValueError(f"Expected depth_conf with 4 dims, got shape {depth_conf.shape}")

    B, S, H, W = depth_conf.shape
    if (H % patch_size) or (W % patch_size):
        raise ValueError(f"Spatial size {(H, W)} is not divisible by patch_size={patch_size}")

    flat = depth_conf.view(B * S, 1, H, W)
    pooled = F.avg_pool2d(flat, kernel_size=patch_size, stride=patch_size)
    pooled = pooled.view(B, S, pooled.shape[-2], pooled.shape[-1])
    return pooled


def main(args: argparse.Namespace) -> None:
    result_dir = Path(args.result_dir)
    if not result_dir.exists():
        raise FileNotFoundError(f"Result directory not found: {result_dir}")

    depth_conf_path = result_dir / "depth_conf.pt"
    if not depth_conf_path.exists():
        raise FileNotFoundError(f"{depth_conf_path} is missing; run Test/inference.py first.")

    depth_conf = torch.load(depth_conf_path)

    metadata = {}
    result_files = sorted(result_dir.glob("inference_result_seq_*.pth"))
    if result_files:
        metadata = torch.load(result_files[0]).get("metadata", {})

    patch_size = args.patch_size or metadata.get("patch_size")
    if patch_size is None:
        patch_size = 14  # fallback

    pooled_conf = pool_confidence(depth_conf, patch_size)
    output_path = result_dir / "depth_conf_patch.pt"
    torch.save(pooled_conf, output_path)

    print(f"Pooled VGGT confidence maps with patch_size={patch_size}")
    print(f"Input shape : {tuple(depth_conf.shape)}")
    print(f"Pooled shape: {tuple(pooled_conf.shape)}")
    print(f"Saved to    : {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Average-pool VGGT confidence maps on patch grid")
    parser.add_argument(
        "--result_dir",
        type=str,
        default="./inference_outputs/botanical_garden",
        help="Directory containing inference outputs",
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=None,
        help="Patch size override (default: take from metadata or 14)",
    )
    main(parser.parse_args())
