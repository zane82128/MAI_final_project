import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from rich.progress import track

from Data.Interface import MVSDataset, MVSData
from Network.vggt import VGGT

from ..metrics.memory import memory_monitor
from ..metrics.pointcloud import evaluation_pipeline


@dataclass
class EvaluateConfig:
    infer_shape: tuple[int, int]


VGGTLike = VGGT
MaskGetter = Callable[[], torch.Tensor | None] | None


def _resize_to_gt(tensor: torch.Tensor, target_hw: tuple[int, int]) -> torch.Tensor:
    """Resize (B, S, C, H, W) tensor to GT spatial resolution."""
    B, S = tensor.shape[:2]
    flat = tensor.flatten(0, 1)
    resized = torch.nn.functional.interpolate(
        flat, size=target_hw, mode="bilinear", align_corners=False
    )
    return resized.view(B, S, *resized.shape[1:])


def _mask_tokens_to_eval(
    mask: torch.Tensor,
    batch_size: int,
    seq_len: int,
    infer_hw: tuple[int, int],
    gt_hw: tuple[int, int],
) -> torch.Tensor:
    """Convert (BS, tokens) mask into (B, S, H_gt, W_gt) boolean mask."""
    patch_h = infer_hw[0] // 14
    patch_w = infer_hw[1] // 14
    expected_tokens = patch_h * patch_w

    mask = mask.view(batch_size * seq_len, -1)
    if mask.shape[1] != expected_tokens:
        raise ValueError(
            f"Unexpected mask token count {mask.shape[1]}, expected {expected_tokens}"
        )

    pixel_mask = mask.view(batch_size, seq_len, 1, patch_h, patch_w).float()
    pixel_mask = pixel_mask.repeat_interleave(14, dim=-2).repeat_interleave(14, dim=-1)
    pixel_mask = pixel_mask.view(batch_size * seq_len, 1, patch_h * 14, patch_w * 14)

    resized = torch.nn.functional.interpolate(pixel_mask, size=gt_hw, mode="nearest")
    resized = resized.view(batch_size, seq_len, 1, *gt_hw).squeeze(2)

    return resized.bool()


def _prepare_eval_mask(
    idx: int,
    cache_dir: Path | None,
    infer_hw: tuple[int, int],
    gt_hw: tuple[int, int],
    gt_depth: torch.Tensor,
    mask_tensor: torch.Tensor | None,
    batch_size: int,
    seq_len: int,
) -> torch.Tensor | None:
    """Load/save mask cache, convert to evaluation pixel mask."""

    def mask_path(i: int) -> Path:
        assert cache_dir is not None
        return cache_dir / f"{i:05d}.pth"

    if mask_tensor is not None and cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        torch.save(mask_tensor.cpu(), mask_path(idx))
    elif mask_tensor is None and cache_dir is not None:
        cached = mask_path(idx)
        if cached.exists():
            mask_tensor = torch.load(cached).to(gt_depth.device)

    if mask_tensor is None:
        return None

    eval_mask = _mask_tokens_to_eval(
        mask_tensor, batch_size, seq_len, infer_hw, gt_hw
    )
    valid_depth = (gt_depth > 0.0) & (gt_depth < 100.0)
    return eval_mask & valid_depth.unsqueeze(0).expand_as(eval_mask)


def _warmup_model(
    model: VGGTLike,
    sample: MVSData,
    config: EvaluateConfig,
    get_mask: MaskGetter,
):
    """Run a single forward pass to warm up CUDA kernels (not timed)."""
    model.eval().cuda()
    images = sample.images.to("cuda")
    B, S = images.shape[:2]

    infer_images = torch.nn.functional.interpolate(
        images.flatten(0, 1),
        size=config.infer_shape,
        mode="bilinear",
        align_corners=False,
    ).view(B, S, 3, *config.infer_shape)

    with torch.no_grad():
        _ = model(infer_images)
        if get_mask is not None:
            _ = get_mask()
        torch.cuda.synchronize()


@torch.no_grad()
def pointcloud_evaluate(
    model: VGGTLike,
    data: MVSDataset,
    config: EvaluateConfig,
    save_result: Path,
    mask_dir: Path | None,
    get_mask: MaskGetter,
    latency_ref: float | None = None,
):
    """Evaluate VGGT variants on point cloud metrics."""
    if save_result.exists():
        print(f"[PointCloud] Skip evaluation because {save_result} already exists.")
        return

    save_result.parent.mkdir(parents=True, exist_ok=True)
    if mask_dir is not None:
        mask_dir.mkdir(parents=True, exist_ok=True)

    torch.cuda.empty_cache()
    model.eval().cuda()

    def run_model(images: torch.Tensor) -> tuple[dict[str, torch.Tensor], float]:
        with memory_monitor():
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record(torch.cuda.current_stream())
            predictions = model(images)
            end_event.record(torch.cuda.current_stream())

            torch.cuda.synchronize()

        runtime = start_event.elapsed_time(end_event)
        return predictions, runtime

    # Warm-up (avoid counting initialization latency)
    try:
        warm_sample = data[0]  # type: ignore[index]
    except Exception:
        warm_sample = None

    if warm_sample is not None:
        print("[PointCloud] Performing warm-up forward pass (excluded from latency).")
        _warmup_model(model, warm_sample, config, get_mask)

    results: list[dict] = []

    for idx, sample in track(
        enumerate(data), total=len(data), description=f"{save_result}"
    ):
        assert isinstance(sample, MVSData)
        B, S = sample.images.shape[:2]
        gt_hw = sample.gt_depths.shape[-2:]

        images = sample.images.to("cuda")
        gt_depths = sample.gt_depths.to("cuda")
        intrinsics = sample.intrinsics.to("cuda")
        poses = sample.poses.to("cuda")
        gt_pointclouds = sample.points[0].to("cuda")

        infer_images = torch.nn.functional.interpolate(
            images.flatten(0, 1),
            size=config.infer_shape,
            mode="bilinear",
            align_corners=False,
        ).view(B, S, 3, *config.infer_shape)

        predictions, runtime_ms = run_model(infer_images)

        depth = predictions["depth"].permute(0, 1, 4, 2, 3)
        world_points = predictions["world_points"].permute(0, 1, 4, 2, 3)

        depth = _resize_to_gt(depth, gt_hw)[0]
        pred_pointclouds = _resize_to_gt(world_points, gt_hw)[0]

        mask_tensor = get_mask() if get_mask is not None else None
        eval_mask = _prepare_eval_mask(
            idx,
            mask_dir,
            config.infer_shape,
            gt_hw,
            gt_depths[0, :, 0],
            mask_tensor,
            B,
            S,
        )
        eval_mask_sample = (
            eval_mask[0] if eval_mask is not None else None
        )

        metrics = evaluation_pipeline(
            pred_pointclouds=pred_pointclouds,
            gt_depths=gt_depths[0],
            gt_pointclouds=gt_pointclouds,
            intrinsics=intrinsics[0],
            poses=poses[0],
            mask=eval_mask_sample,
        )

        entry = {
            "runtime_ms": runtime_ms,
            "accuracy": metrics.accuracy,
            "completeness": metrics.completeness,
            "chamfer_distance": metrics.chamfer_distance,
        }
        results.append(entry)
        print(
            f"[{idx:04d}/{len(data)}] runtime={runtime_ms:.2f} ms | "
            f"comp={metrics.completeness:.4f} | acc={metrics.accuracy:.4f}"
        )

    with open(save_result, "w") as f:
        json.dump(results, f)

    runtimes = torch.tensor([r["runtime_ms"] for r in results], dtype=torch.float64)
    completeness = torch.tensor(
        [r["completeness"] for r in results], dtype=torch.float64
    )
    accuracy = torch.tensor([r["accuracy"] for r in results], dtype=torch.float64)

    avg_latency = float(runtimes.mean()) if len(runtimes) else float("nan")
    avg_comp = float(completeness.mean()) if len(completeness) else float("nan")
    avg_acc = float(accuracy.mean()) if len(accuracy) else float("nan")
    speedup = (
        (latency_ref / avg_latency) if latency_ref and avg_latency else None
    )

    print("\n[PointCloud Summary]")
    print(f"  Samples     : {len(results)}")
    print(f"  Latency (ms): {avg_latency:.2f}")
    if speedup is not None:
        print(f"  Speedup     : {speedup:.2f}x (ref={latency_ref:.2f} ms)")
    print(f"  Completeness: {avg_comp:.4f}")
    print(f"  Accuracy    : {avg_acc:.4f}")
