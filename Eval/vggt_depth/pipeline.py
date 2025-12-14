import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable

import torch
from rich.progress import track

from Data.Interface import DepthDataset, DepthData
from Network.vggt import VGGT

from ..metrics.depth import evaluation_pipeline
from ..metrics.memory import memory_monitor


@dataclass
class EvalauteConfig:
    infer_shape: tuple[int, int]
    dataset_name: str | None = None


VGGTLike = VGGT
MaskGetter = Callable[[], torch.Tensor | None] | None


def _resize_to_gt(tensor: torch.Tensor, target_hw: tuple[int, int]) -> torch.Tensor:
    """Resize (B, S, C, H, W) tensor to GT spatial resolution."""
    B, S = tensor.shape[:2]
    flat = tensor.flatten(0, 1)
    resized = torch.nn.functional.interpolate(
        flat, size=target_hw, mode="bicubic", align_corners=False
    )
    return resized.view(B, S, *resized.shape[1:])


def _mask_tokens_to_eval(
    mask: torch.Tensor, infer_hw: tuple[int, int], gt_hw: tuple[int, int]
) -> torch.Tensor:
    """Convert (N, tokens) mask into pixel-level boolean mask."""
    patch_h = infer_hw[0] // 14
    patch_w = infer_hw[1] // 14
    expected = patch_h * patch_w

    mask = mask.view(-1, expected)
    pixel = mask.view(-1, 1, patch_h, patch_w).float()
    pixel = pixel.repeat_interleave(14, dim=-2).repeat_interleave(14, dim=-1)
    resized = torch.nn.functional.interpolate(pixel, size=gt_hw, mode="nearest")
    return resized.squeeze(1).bool()


def _prepare_eval_mask(
    idx: int,
    cache_dir: Path | None,
    infer_hw: tuple[int, int],
    gt_hw: tuple[int, int],
    gt_depth: torch.Tensor,
    mask_tensor: torch.Tensor | None,
) -> torch.Tensor | None:
    def mask_path(i: int) -> Path:
        assert cache_dir is not None
        return cache_dir / f"{i:05d}.pth"

    if mask_tensor is not None and cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        torch.save(mask_tensor.cpu(), mask_path(idx))
    elif mask_tensor is None and cache_dir is not None:
        cached = mask_path(idx)
        if cached.exists():
            mask_tensor = torch.load(cached, map_location="cuda")

    if mask_tensor is None:
        return None

    eval_mask = _mask_tokens_to_eval(mask_tensor, infer_hw, gt_hw)
    eval_mask = eval_mask.view(gt_depth.shape[0], *gt_hw)
    valid_depth = (gt_depth > 0.0) & (gt_depth < 100.0)
    return eval_mask & valid_depth


@torch.no_grad()
def depth_evaluate(
    model: VGGTLike,
    data: DepthDataset,
    config: EvalauteConfig,
    save_result: Path,
    save_mask: Path | None,
    get_mask: MaskGetter,
):
    if save_result.exists():
        print(f"[Depth] Skip evaluation because {save_result} already exists.")
        return

    save_result.parent.mkdir(parents=True, exist_ok=True)
    if save_mask is not None:
        save_mask.mkdir(parents=True, exist_ok=True)

    torch.cuda.empty_cache()
    model.eval().cuda()

    def infer_model(images: torch.Tensor):
        assert model.depth_head is not None
        aggregated_tokens_list, ps_idx = model.aggregator(images)
        depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images, ps_idx)
        return depth_map.permute(0, 1, 4, 2, 3), depth_conf

    def benchmark_model(images: torch.Tensor) -> tuple[float, float]:
        assert model.depth_head is not None
        with memory_monitor() as mem_stat:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record(torch.cuda.current_stream())
            tokens, ps_idx = model.aggregator(images)
            model.depth_head(tokens, images, ps_idx)
            end_event.record(torch.cuda.current_stream())

            torch.cuda.synchronize()

        mem_usage = mem_stat.peak_increased_gb
        assert mem_usage is not None
        return start_event.elapsed_time(end_event), mem_usage

    results = []

    for idx, sample in track(
        enumerate(data), total=len(data), description=f"{save_result}"
    ):
        sample = sample  # type: ignore[assignment]
        assert isinstance(sample, DepthData)
        B, S = sample.images.shape[:2]
        gt_hw = sample.gt_depths.shape[-2:]

        images = sample.images.to("cuda")
        gt_depths = sample.gt_depths.to("cuda")

        infer_images = torch.nn.functional.interpolate(
            images.flatten(0, 1),
            size=config.infer_shape,
            mode="bilinear",
            align_corners=False,
        ).view(B, S, 3, *config.infer_shape)

        depth, depth_conf = infer_model(infer_images)
        runtime_ms, memory_gb = benchmark_model(infer_images)

        depth = _resize_to_gt(depth, gt_hw)[0]  # (S, 1, H, W)
        depth = depth[:, 0]

        inference_mask = get_mask() if get_mask is not None else None
        eval_mask = _prepare_eval_mask(
            idx,
            save_mask,
            config.infer_shape,
            gt_hw,
            gt_depths[0, :, 0],
            inference_mask,
        )
        eval_mask_sample = (
            eval_mask if eval_mask is None else eval_mask.view(S, *gt_hw)
        )

        metrics = evaluation_pipeline(
            pred_depth=depth,
            gt_depth=gt_depths[0, :, 0],
            mask=eval_mask_sample,
        )

        results.append(
            {
                "runtime_ms": runtime_ms,
                "memory_gb": memory_gb,
                "avg_l1": metrics.avg_l1,
                "avg_rmse": metrics.avg_rmse,
                "avg_absrel": metrics.avg_absrel,
                "avg_δ1_25": metrics.avg_δ1_25,
            }
        )

    with open(save_result, "w") as f:
        json.dump(results, f)
