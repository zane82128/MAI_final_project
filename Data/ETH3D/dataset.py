import logging
import numpy as np
import torch
import time
from pathlib import Path
from typing import Optional, Union, Callable
from torchvision.io import decode_image

from ..Interface import DepthData, DepthDataset


logger = logging.getLogger(__name__)


class ETH3D_Depth_Dataset(DepthDataset):
    SCENE_NAMES = ["courtyard", "delivery_area", "electro", "facade", "kicker", "meadow", "office", "pipes", "playground", "relief", "relief_2", "terrace", "terrains"]
    
    def __init__(self, root: str | Path, sequence_length: int):
        self.root = Path(root)
        self.seql = sequence_length
        
        self.samples = self._sample_all()
    
    def _sample_all(self) -> list[list[tuple[Path, Path]]]:
        scenes_paths = [Path(self.root, scene) for scene in self.SCENE_NAMES]
        
        samples = []
        total_scenes = len(scenes_paths)
        dataset_start = time.perf_counter()
        print(f"[ETH3D_Depth] Sampling sequences from {total_scenes} scenes under {self.root}")
        for scene_idx, scene_path in enumerate(scenes_paths, start=1):
            scene_start = time.perf_counter()
            depth_root = Path(scene_path, "ground_truth_depth", "dslr_images")
            image_root = Path(scene_path, "images", "dslr_images")
            
            depth_files  = sorted(list(depth_root.rglob("*.JPG")))
            image_files  = sorted(list(image_root.rglob("*.JPG")))

            common_names = {d.name for d in depth_files} & {i.name for i in image_files}
            common_pred  = lambda x: x.name in common_names
            pairs        = list(zip(filter(common_pred, depth_files), filter(common_pred, image_files)))
            
            before = len(samples)
            for start_idx in range(0, len(pairs) - self.seql, self.seql):
                samples.append(pairs[start_idx:start_idx+self.seql])
            scene_sequences = len(samples) - before
            elapsed_scene = time.perf_counter() - scene_start
            elapsed_total = time.perf_counter() - dataset_start
            print(
                f"[ETH3D_Depth] {scene_path.name:<15} -> +{scene_sequences:3d} sequences "
                f"({len(samples)} total) | scene {scene_idx}/{total_scenes} | "
                f"{elapsed_scene:5.1f}s scene / {elapsed_total:6.1f}s total"
            )
        
        print(f"[ETH3D_Depth] Prepared {len(samples)} sequences in {time.perf_counter() - dataset_start:.1f}s")
        return samples

    def __len__(self) -> int: return len(self.samples)
    
    def __getitem__(self, index) -> DepthData:
        sample = self.samples[index]
        
        depth = torch.stack(
            [self._load_depth(d) for d, i in sample], dim=0
        )
        image = torch.stack(
            [self._load_image(i) for d, i in sample], dim=0
        )
        return DepthData(images=image.unsqueeze(0), gt_depths=depth.unsqueeze(0))

    def _load_image(self, image_path: Path) -> torch.Tensor:
        """Load and preprocess a single image."""
        try:
            image = decode_image(str(image_path)).float() / 255.0

            if image.shape[0] == 1:
                image = image.repeat(3, 1, 1)
            elif image.shape[0] == 4:
                image = image[:3]

            return image
        except Exception as e:
            raise RuntimeError(f"Failed to load image {image_path}: {e}")

    def _load_depth(self, depth_path: Path) -> torch.Tensor:
        """Load and preprocess a single depth map."""
        try:
            depth_data = np.fromfile(depth_path, dtype=np.float32)
            depth = torch.from_numpy(depth_data.reshape(4032, 6048)).float()
            depth = depth.unsqueeze(0)

            return depth
        except Exception as e:
            raise RuntimeError(f"Failed to load depth {depth_path}: {e}")
