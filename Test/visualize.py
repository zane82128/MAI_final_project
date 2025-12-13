"""
Visualize VGGT confidence maps and CoMe masks.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap
from PIL import Image

YELLOW_BLACK_CMAP = ListedColormap(["#000000", "#FFD400"])


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _extract_frames(tensor: torch.Tensor) -> torch.Tensor:
    """Return a tensor shaped as [S, H, W] for per-frame processing."""
    if tensor.ndim == 4:  # [B, S, H, W]
        return tensor[0]
    if tensor.ndim == 3:  # [S, H, W]
        return tensor
    if tensor.ndim == 2:  # [H, W] -> single frame
        return tensor.unsqueeze(0)
    if tensor.ndim == 1:  # [N] flattened, treat as single frame
        return tensor.unsqueeze(0)
    raise ValueError(f"Unsupported tensor shape: {tensor.shape}")


def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def _resize_mask(mask: np.ndarray, target_shape: tuple[int, int] | None) -> np.ndarray:
    if target_shape is None or mask.shape == target_shape:
        return mask

    pil = Image.fromarray(mask.astype(np.uint8))
    pil = pil.resize((target_shape[1], target_shape[0]), resample=Image.NEAREST)
    return np.array(pil)


def load_results(result_dir: Path):
    """Load inference tensors and metadata."""
    depth_conf = torch.load(result_dir / "depth_conf.pt")
    patch_conf = None
    patch_conf_path = result_dir / "depth_conf_patch.pt"
    if patch_conf_path.exists():
        patch_conf = torch.load(patch_conf_path)

    come_mask = None
    mask_path = result_dir / "come_mask.pt"
    if mask_path.exists():
        come_mask = torch.load(mask_path)

    result_files = sorted(result_dir.glob("inference_result_seq_*.pth"))
    if not result_files:
        raise FileNotFoundError("Missing inference_result_seq_*.pth")
    full_result = torch.load(result_files[0])
    metadata = full_result.get("metadata", {})

    return depth_conf, come_mask, patch_conf, metadata


def visualize_confidence_map(
    depth_conf: torch.Tensor,
    output_dir: Path,
    title_prefix: str = "VGGT Confidence Map",
    frame_indices: list[int] | None = None,
) -> list[Path]:
    """Render a confidence map for each frame."""
    _ensure_dir(output_dir)
    frames = _extract_frames(depth_conf)
    total_frames = frames.shape[0]
    indices = frame_indices or list(range(total_frames))

    saved_paths: list[Path] = []
    for idx in indices:
        if idx >= total_frames:
            continue
        conf = _to_numpy(frames[idx])
        conf_min, conf_max = conf.min(), conf.max()
        conf_normalized = (conf - conf_min) / (conf_max - conf_min + 1e-8)

        fig, ax = plt.subplots(figsize=(8, 6), dpi=120)
        im = ax.imshow(conf_normalized, cmap="jet", origin="upper")
        ax.set_title(f"{title_prefix} #{idx:02d}", fontsize=13, fontweight="bold")
        plt.colorbar(im, ax=ax, label=f"Confidence [{conf_min:.3f} - {conf_max:.3f}]")
        ax.set_xlabel("Width")
        ax.set_ylabel("Height")

        out_path = output_dir / f"confidence_map_{idx:02d}.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(out_path)

    return saved_paths


def visualize_mask(
    come_mask: torch.Tensor | None,
    output_dir: Path,
    target_shape: tuple[int, int] | None = None,
    frame_indices: list[int] | None = None,
) -> list[Path]:
    """Render the CoMe mask per frame."""
    if come_mask is None:
        print("Warning: CoMe mask is None, skipping visualization")
        return []
    if isinstance(come_mask, dict):
        print(f"Warning: CoMe mask is a dict and cannot be visualized directly: {come_mask.keys()}")
        return []

    _ensure_dir(output_dir)
    frames = _extract_frames(come_mask)
    total_frames = frames.shape[0]
    indices = frame_indices or list(range(total_frames))

    saved_paths: list[Path] = []
    for idx in indices:
        if idx >= total_frames:
            continue
        mask = _to_numpy(frames[idx])
        mask = _resize_mask(mask, target_shape)

        fig, ax = plt.subplots(figsize=(8, 6), dpi=120)
        if mask.dtype in [bool, np.bool_]:
            data = mask.astype(np.uint8)
            im = ax.imshow(data, cmap=YELLOW_BLACK_CMAP, origin="upper", vmin=0, vmax=1)
            ax.set_title(f"CoMe Mask #{idx:02d} (Yellow=Keep)", fontsize=13, fontweight="bold")
            cbar = plt.colorbar(im, ax=ax, ticks=[0, 1])
            cbar.ax.set_yticklabels(["Merge", "Keep"])
        else:
            im = ax.imshow(mask, cmap="viridis", origin="upper")
            ax.set_title(f"CoMe Mask #{idx:02d}", fontsize=13, fontweight="bold")
            plt.colorbar(im, ax=ax, label="Mask Value")

        ax.set_xlabel("Width")
        ax.set_ylabel("Height")
        out_path = output_dir / f"come_mask_{idx:02d}.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(out_path)

    return saved_paths


def create_comparison_figure(
    depth_conf: torch.Tensor,
    come_mask: torch.Tensor | None,
    output_dir: Path,
    metadata: dict,
    frame_idx: int,
    mask_target_shape: tuple[int, int] | None = None,
) -> Path:
    """Create a side-by-side visualization of confidence map and mask."""
    _ensure_dir(output_dir)
    frames_conf = _extract_frames(depth_conf)
    if frame_idx >= frames_conf.shape[0]:
        raise IndexError(f"Frame index {frame_idx} is out of range")

    conf = _to_numpy(frames_conf[frame_idx])
    conf_norm = (conf - conf.min()) / (conf.max() - conf.min() + 1e-8)

    mask_img = None
    if come_mask is not None and not isinstance(come_mask, dict):
        frames_mask = _extract_frames(come_mask)
        if frame_idx < frames_mask.shape[0]:
            mask_np = _to_numpy(frames_mask[frame_idx])
            mask_img = _resize_mask(mask_np, mask_target_shape)

    cols = 2 if mask_img is not None else 1
    fig, axes = plt.subplots(1, cols, figsize=(14 if cols == 2 else 7, 6), dpi=120)
    if cols == 1:
        axes = [axes]

    im0 = axes[0].imshow(conf_norm, cmap="jet", origin="upper")
    axes[0].set_title(f"VGGT Confidence #{frame_idx:02d}", fontsize=12, fontweight="bold")
    plt.colorbar(im0, ax=axes[0], label="Confidence")

    if mask_img is not None and cols == 2:
        im1 = axes[1].imshow(mask_img.astype(np.uint8), cmap=YELLOW_BLACK_CMAP, origin="upper", vmin=0, vmax=1)
        axes[1].set_title(f"CoMe Mask #{frame_idx:02d} (Yellow=Keep)", fontsize=12, fontweight="bold")
        cbar = plt.colorbar(im1, ax=axes[1], ticks=[0, 1])
        cbar.ax.set_yticklabels(["Merge", "Keep"])

    fig.suptitle(
        f"Scene: {metadata.get('scene', 'Unknown')} | Seq Length: {metadata.get('sequence_length', '?')} | Frame #{frame_idx}",
        fontsize=11,
    )
    out_path = output_dir / f"comparison_{frame_idx:02d}.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main(args):
    result_dir = Path(args.result_dir)
    if not result_dir.exists():
        raise FileNotFoundError(f"Result directory not found: {result_dir}")

    print(f"Loading results from {result_dir}")
    depth_conf, come_mask, patch_conf, metadata = load_results(result_dir)
    print(f"Confidence map shape: {tuple(depth_conf.shape)}")
    if come_mask is not None:
        print(f"CoMe mask shape: {tuple(come_mask.shape)}")
    else:
        print("Warning: come_mask.pt not found; only confidence maps will be generated")
    if patch_conf is not None:
        print(f"Patch-averaged confidence shape: {tuple(patch_conf.shape)}")
    else:
        print("Note: depth_conf_patch.pt not found; run Test/pool_confidence.py to generate patch-level tensors")

    frames_conf = _extract_frames(depth_conf)
    seq_len = frames_conf.shape[0]
    target_shape = frames_conf.shape[1:]  # (H, W)

    viz_dir = result_dir / "visualizations"
    conf_dir = viz_dir / "confidence_maps"
    pooled_conf_dir = viz_dir / "confidence_maps_patch"
    mask_dir = viz_dir / "come_masks"
    cmp_dir = viz_dir / "comparisons"

    print("\nGenerating confidence maps...")
    conf_paths = visualize_confidence_map(depth_conf, conf_dir, title_prefix="VGGT Confidence Map")
    print(f"   -> Saved {len(conf_paths)} images")

    pooled_paths: list[Path] = []
    if patch_conf is not None:
        print("Generating patch-averaged confidence maps...")
        pooled_paths = visualize_confidence_map(
            patch_conf,
            pooled_conf_dir,
            title_prefix="Patch-Averaged VGGT Confidence",
        )
        print(f"   -> Saved {len(pooled_paths)} images")

    mask_paths: list[Path] = []
    if come_mask is not None and not isinstance(come_mask, dict):
        print("Generating CoMe masks...")
        mask_paths = visualize_mask(come_mask, mask_dir, target_shape=target_shape)
        print(f"   -> Saved {len(mask_paths)} images")

    print("Generating confidence vs. mask comparisons...")
    _ensure_dir(cmp_dir)
    for idx in range(seq_len):
        create_comparison_figure(
            depth_conf,
            come_mask if (come_mask is not None and not isinstance(come_mask, dict)) else None,
            cmp_dir,
            metadata,
            frame_idx=idx,
            mask_target_shape=target_shape,
        )

    print(f"\nDone. Outputs stored in {viz_dir}")
    print(f"   - {conf_dir}")
    if pooled_paths:
        print(f"   - {pooled_conf_dir}")
    if mask_paths:
        print(f"   - {mask_dir}")
    print(f"   - {cmp_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize CoMe inference outputs (per frame)")
    parser.add_argument(
        "--result_dir",
        type=str,
        default="./inference_outputs/botanical_garden",
        help="Directory containing inference outputs (default: ./inference_outputs/botanical_garden)",
    )
    main(parser.parse_args())
