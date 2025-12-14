from pathlib import Path

from Data.Interface import MVSDataset

from .pipeline import EvaluateConfig, pointcloud_evaluate
from ..common.datautil import slice_dataset


def get_dataset(args) -> MVSDataset:
    match args.dataset:
        case "eth-3d":
            from Data.ETH3D import ETH3D_MVS_Dataset, default_root

            dataset = ETH3D_MVS_Dataset(default_root, args.seq_l)
        case "dtu-mvs":
            from Data.DTU import DTU_MVSDataset, default_root

            dataset = DTU_MVSDataset(default_root, args.seq_l)
        case _:
            raise ValueError(f"Unsupported dataset {args.dataset}")

    return slice_dataset(dataset, args.slice)


def get_model(args):
    match args.model:
        case "vggt":
            from Network.vggt import get_VGGT_vanilla

            return get_VGGT_vanilla(), None
        case "vggt*":
            from Network.vggt import get_VGGT

            return get_VGGT(), None
        case "ours":
            from Network.vggt import get_VGGT
            from Accelerate.vggt import Accelerator_Config, accelerate_vggt

            mask_setup = (args.mask_setup[0], float(args.mask_setup[1]))
            model = get_VGGT()
            model, get_mask, _ = accelerate_vggt(
                model,
                Accelerator_Config(
                    accelerator=args.accelerator,
                    grp_size=args.grp_size,
                    mask_setup=mask_setup,
                    accel_dino_attn=not args.ablation_no_dino_att_merge,
                    accel_dino_mlp=not args.ablation_no_dino_mlp_merge,
                    accel_frame_attn=not args.ablation_no_frame_att_merge,
                    accel_frame_mlp=not args.ablation_no_frame_mlp_merge,
                    accel_global_attn=not args.ablation_no_globe_att_merge,
                    accel_global_mlp=not args.ablation_no_globe_mlp_merge,
                    apply_attn_bias=not args.ablation_no_attn_bias,
                ),
            )
            return model, get_mask
        case other:
            raise ValueError(f"Unsupported model type {other}")


def main(args):
    dataset = get_dataset(args)
    model, get_mask = get_model(args)

    pointcloud_evaluate(
        model,
        dataset,
        EvaluateConfig(infer_shape=(27 * 14, 36 * 14)),
        args.result,
        args.masks,
        get_mask,
        latency_ref=args.latency_ref,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Point cloud evaluation for VGGT and Co-Me variants"
    )

    group = parser.add_argument_group("Dataset")
    group.add_argument(
        "--dataset", type=str, choices=["eth-3d", "dtu-mvs"], required=True
    )
    group.add_argument(
        "--seq_l", type=int, metavar="S", required=True, help="Sequence length"
    )
    group.add_argument(
        "--slice",
        type=int,
        metavar="N",
        default=None,
        help="Evaluate only the first N samples",
    )

    subparsers = parser.add_subparsers(dest="model", required=True)
    subparsers.add_parser("vggt", help="Vanilla VGGT")
    subparsers.add_parser("vggt*", help="VGGT* (VGGT + FlexAttention)")

    ours_parser = subparsers.add_parser("ours", help="Co-Me accelerated VGGT")
    ours_parser.add_argument("--grp_size", type=int, required=True)
    ours_parser.add_argument(
        "--mask_setup",
        nargs=2,
        metavar=("Method", "Threshold"),
        required=True,
        help="('z-score', float) | ('bot-k', int) | ('bot-p', float)",
    )
    ours_parser.add_argument(
        "--accelerator",
        type=Path,
        required=True,
        help="Path to VGGT accelerator checkpoint",
    )
    ours_parser.add_argument("--ablation_no_frame_att_merge", action="store_true")
    ours_parser.add_argument("--ablation_no_frame_mlp_merge", action="store_true")
    ours_parser.add_argument("--ablation_no_globe_att_merge", action="store_true")
    ours_parser.add_argument("--ablation_no_globe_mlp_merge", action="store_true")
    ours_parser.add_argument("--ablation_no_dino_att_merge", action="store_true")
    ours_parser.add_argument("--ablation_no_dino_mlp_merge", action="store_true")
    ours_parser.add_argument("--ablation_no_attn_bias", action="store_true")

    parser.add_argument(
        "--result", type=Path, required=True, help="Output JSON for metrics"
    )
    parser.add_argument(
        "--masks",
        type=Path,
        default=None,
        help="Directory to cache/load Co-Me masks (ours only)",
    )
    parser.add_argument(
        "--latency_ref",
        type=float,
        default=None,
        help="Reference latency (ms) for speedup reporting",
    )

    main(parser.parse_args())
