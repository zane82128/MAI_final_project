import argparse
import torch
import time
import traceback
import sys
from pathlib import Path
from torchvision.io import decode_image
from Network.vggt import get_VGGT
from Accelerate.vggt import accelerate_vggt, Accelerator_Config

parser = argparse.ArgumentParser(description="Run VGGT + CoMe inference on an ETH3D scene.")
parser.add_argument("--scene", type=str, default="botanical_garden", help="ETH3D scene name under datasets/ETH3D/")
parser.add_argument("--dataset_root", type=Path, default=Path("./datasets/ETH3D"), help="Root to ETH3D dataset.")
parser.add_argument("--image_dir", type=Path, default=None, help="Override image directory path.")
parser.add_argument("--seq_l", type=int, default=16, help="Sequence length.")
parser.add_argument("--infer_height", type=int, default=27 * 14, help="Inference height (must be divisible by 14).")
parser.add_argument("--infer_width", type=int, default=36 * 14, help="Inference width (must be divisible by 14).")
parser.add_argument("--accelerator", type=Path, default=Path("./Model/VGGT_Accelerator/checkpoint.pth"))
parser.add_argument("--result_dir", type=Path, default=None, help="Directory where outputs are saved. Defaults to ./inference_outputs/<scene>.")
parser.add_argument("--patch_size", type=int, default=14)
args = parser.parse_args()

# ============= Configuration (derived) =============
if args.image_dir is not None:
    image_dir = args.image_dir
else:
    image_dir = args.dataset_root / args.scene / "images" / "dslr_images"

seq_l = args.seq_l
infer_shape = (args.infer_height, args.infer_width)
accelerator_path = args.accelerator
output_dir = args.result_dir if args.result_dir is not None else Path("./inference_outputs") / args.scene
output_dir.mkdir(parents=True, exist_ok=True)
patch_size = args.patch_size  # VGGT uses a fixed 14x14 patch size internally


total_start = time.time()

try:
    # ============= Step 1: Prepare image sequence =============
    print(f"[1/5] Preparing image sequence", flush=True)
    step1_start = time.time()
    
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    
    image_files = sorted(list(image_dir.rglob("*.JPG")))
    print(f"      Found {len(image_files)} images", flush=True)
    
    if len(image_files) < seq_l:
        raise ValueError(f"Number of images ({len(image_files)}) < sequence length ({seq_l})")
    
    image_files = image_files[:seq_l]
    print(f"      Using the first {seq_l} images", flush=True)
    print(f"      Elapsed: {time.time() - step1_start:.2f}s\n", flush=True)
    
    # ============= Step 2: Load images =============
    print(f"[2/5] Loading images", flush=True)
    step2_start = time.time()
    
    def load_image(image_path: Path) -> torch.Tensor:
        image = decode_image(str(image_path)).float() / 255.0
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)
        elif image.shape[0] == 4:
            image = image[:3]
        return image
    
    frames = []
    orig_H, orig_W = None, None
    for i, image_path in enumerate(image_files):
        if i % 4 == 0:
            print(f"      Loading progress: {i}/{len(image_files)}", flush=True)
        frame = load_image(image_path)
        frames.append(frame)
        if orig_H is None:
            orig_H, orig_W = frame.shape[1:]
            print(f"      Original image size: {orig_W}x{orig_H}", flush=True)
    
    print(f"      Elapsed: {time.time() - step2_start:.2f}s\n", flush=True)
    
    # ============= Step 3: Build input tensor =============
    print(f"[3/5] Building input tensor", flush=True)
    step3_start = time.time()
    
    images = torch.stack(frames).unsqueeze(0)
    print(f"      Stacked tensor shape: {tuple(images.shape)}", flush=True)
    
    images = torch.nn.functional.interpolate(
        images.flatten(0, 1), size=infer_shape, mode='bilinear', align_corners=False
    ).view(1, seq_l, 3, *infer_shape)
    
    print(f"      Resized tensor shape: {tuple(images.shape)}", flush=True)
    print(f"      Elapsed: {time.time() - step3_start:.2f}s\n", flush=True)
    
    # ============= Step 4: Load model =============
    print(f"[4/5] Loading VGGT model and CoMe accelerator", flush=True)
    step4_start = time.time()
    
    print(f"      [4.1] Loading VGGT base model...", flush=True)
    model_start = time.time()
    model = get_VGGT()
    print(f"            Done ({time.time() - model_start:.2f}s)", flush=True)
    
    print(f"      [4.2] Loading CoMe accelerator...", flush=True)
    if not accelerator_path.exists():
        raise FileNotFoundError(f"Accelerator checkpoint not found: {accelerator_path}")
    
    accel_start = time.time()
    model, get_mask, set_mask = accelerate_vggt(
        model,
        Accelerator_Config(
            accelerator=accelerator_path,
            grp_size=4,
            mask_setup=("bot-p", 0.5),
        )
    )
    print(f"            Done ({time.time() - accel_start:.2f}s)", flush=True)
    
    print(f"      [4.3] Moving model to GPU...", flush=True)
    gpu_start = time.time()
    model.cuda().eval()
    torch.cuda.synchronize()
    print(f"            Done ({time.time() - gpu_start:.2f}s)", flush=True)
    print(f"      Elapsed: {time.time() - step4_start:.2f}s\n", flush=True)
    
    # ============= Step 5: Inference =============
    print(f"[5/5] Running inference", flush=True)
    step5_start = time.time()
    
    with torch.no_grad():
        print(f"      Moving input to GPU...", flush=True)
        gpu_input_start = time.time()
        images = images.cuda()
        torch.cuda.synchronize()
        print(f"      Done ({time.time() - gpu_input_start:.2f}s)", flush=True)
        
        print(f"      Executing forward pass...", flush=True)
        forward_start = time.time()
        predictions = model(images)
        torch.cuda.synchronize()
        print(f"      Done ({time.time() - forward_start:.2f}s)", flush=True)
        
        depth = predictions["depth"]
        depth_conf = predictions["depth_conf"]
        world_points = predictions["world_points"]
        world_points_conf = predictions["world_points_conf"]
        
        print(f"      Depth shape: {tuple(depth.shape)}", flush=True)
        print(f"      Depth confidence shape: {tuple(depth_conf.shape)}", flush=True)
        
        print(f"      Fetching CoMe mask...", flush=True)
        mask_start = time.time()
        come_mask = get_mask()
        torch.cuda.synchronize()
        print(f"      Done ({time.time() - mask_start:.2f}s)", flush=True)
        if come_mask is not None:
            print(f"      CoMe mask shape: {tuple(come_mask.shape)}", flush=True)
        else:
            print(f"      Warning: CoMe mask is None", flush=True)
    
    print(f"      Elapsed: {time.time() - step5_start:.2f}s\n", flush=True)
    
    # ============= Save results =============
    print(f"[Save] Writing inference outputs to {output_dir}", flush=True)
    save_start = time.time()
    
    # Derive scene name, defaulting to CLI argument
    scene_name = args.scene or image_dir.parent.parent.name
    result_subdir = output_dir
    
    # Move come_mask to CPU for serialization (if available)
    come_mask_cpu = come_mask.cpu() if come_mask is not None else None
    come_mask_retention_ratio = None
    come_mask_grid_shape = None
    if come_mask_cpu is not None and isinstance(come_mask_cpu, torch.Tensor):
        come_mask_retention_ratio = float(come_mask_cpu.float().mean())
        batch_size = depth.shape[0]
        seq_len = depth.shape[1]
        if infer_shape[0] % patch_size == 0 and infer_shape[1] % patch_size == 0:
            tok_h = infer_shape[0] // patch_size
            tok_w = infer_shape[1] // patch_size
            expected = tok_h * tok_w
            if come_mask.dim() == 2 and come_mask.shape == (batch_size * seq_len, expected):
                come_mask = come_mask.view(batch_size, seq_len, tok_h, tok_w)
                come_mask_grid_shape = (tok_h, tok_w)
                print(f"      Reshaped mask -> {tuple(come_mask.shape)}", flush=True)
            else:
                print(f"      Warning: expected mask shape ({batch_size * seq_len}, {expected})", flush=True)
        else:
            print("      Warning: inference resolution not divisible by patch size, skipping mask reshape", flush=True)


    # Save tensors for later visualization
    result_dict = {
        # ===== Depth prediction =====
        "depth": depth.cpu(),                           # [B, S, H, W, 1]
        "depth_conf": depth_conf.cpu(),                 # [B, S, H, W] ← VGGT Confidence
        
        # ===== 3D point cloud =====
        "world_points": world_points.cpu(),             # [B, S, H, W, 3]
        "world_points_conf": world_points_conf.cpu(),   # [B, S, H, W]
        
        # ===== CoMe mask =====
        "come_mask": come_mask_cpu,  # complex structure or Tensor
        
        # ===== Metadata =====
        "metadata": {
            "scene": scene_name,
            "original_shape": (orig_H, orig_W),
            "infer_shape": infer_shape,
            "sequence_length": seq_l,
            "num_images": len(image_files),
            "image_dir": str(image_dir),
            "depth_conf_range": [float(depth_conf.min()), float(depth_conf.max())],
            "come_mask_retention_ratio": come_mask_retention_ratio,
            "patch_size": patch_size,
            "confidence_patch_grid": (infer_shape[0] // patch_size, infer_shape[1] // patch_size),
            "come_mask_patch_grid": come_mask_grid_shape,
        }
    }
    
    result_file = result_subdir / f"inference_result_seq_{seq_l}.pth"
    torch.save(result_dict, result_file)
    print(f"      Saved aggregated results to: {result_file}", flush=True)
    
    # Also save confidence and mask tensors separately for visualization
    torch.save(depth_conf.cpu(), result_subdir / "depth_conf.pt")
    print(f"      Saved confidence map tensor to: {result_subdir / 'depth_conf.pt'}", flush=True)

    if come_mask is not None:
        torch.save(come_mask.cpu(), result_subdir / "come_mask.pt")
        print(f"      Saved CoMe mask tensor to: {result_subdir / 'come_mask.pt'}", flush=True)

    print(f"      Elapsed: {time.time() - save_start:.2f}s\n", flush=True)
    
    # Print summary statistics
    print(f"[Stats]", flush=True)
    print(f"      Depth range: [{depth.min():.3f}, {depth.max():.3f}]", flush=True)
    print(f"      Depth confidence range: [{depth_conf.min():.3f}, {depth_conf.max():.3f}]", flush=True)
    if come_mask is not None:
        come_mask_cpu = come_mask.cpu()
        print(f"      CoMe mask retention ratio: {come_mask_cpu.float().mean():.1%}", flush=True)

    total_time = time.time() - total_start
    print(f"\nInference finished", flush=True)
    print(f"Total time: {total_time:.2f}s ({int(total_time//60)}m {int(total_time%60)}s)", flush=True)

except Exception as e:
    print(f"\nError: {e}", flush=True)
    print(f"\nTraceback:", flush=True)
    traceback.print_exc()
    sys.exit(1)
