# Point Cloud Evaluation Notes

This log tracks the exact commands and measurements used during ETH3D point-cloud experiments. The environment is set up once via:

```
# start Docker container (CUDA 12.8 image)
cd Env/Linux && bash ./start_interact.sh

# build CUDA extension
bash ./install_CUExt.sh

# project venv
python3 -m venv /workspace/.venv
source /workspace/.venv/bin/activate
pip install --upgrade pip

# core dependencies
pip install --extra-index-url https://download.pytorch.org/whl/cu128 \
    torch==2.7.0+cu128 torchvision==0.22.0+cu128
pip install -r Env/Linux/requirements.txt
pip install --no-build-isolation \
    "pytorch3d@git+https://github.com/facebookresearch/pytorch3d.git@stable"
```

After the above, make sure `/workspace/datasets/ETH3D` contains the dataset. Feel free to change the `seq`/`slc` variables if you want different sequence lengths or slices. Recorded metrics are appended as runs finish; remaining entries can be filled in later.

---

## Generic Command Template
To reproduce “frame = X, slice = Y” for the VGGT baseline, substitute the desired values into `seq` / `slc`. Here `slice` simply means **evaluate only the first Y samples** from the ETH3D dataset—handy when you want a quick sanity check before running the full split.
```
seq=X
slc=Y

python -m Eval.vggt_cloud_point \
  --dataset eth-3d \
  --seq_l "$seq" \
  --slice "$slc" \
  --result "ours_results/eth3d_vggt_pointcloud_seq_${seq}_slc_${slc}.json" \
  --masks "ours_results/eth3d_vggt_masks_seq_${seq}_slc_${slc}" \
  --latency_ref 3190 \
  vggt
```

## Frame 1 / Slice 5

### VGGT Command
```
seq=1
slc=5

python -m Eval.vggt_cloud_point \
  --dataset eth-3d \
  --seq_l "$seq" \
  --slice "$slc" \
  --result "ours_results/eth3d_vggt_pointcloud_seq_${seq}_slc_${slc}.json" \
  --masks "ours_results/eth3d_vggt_masks_seq_${seq}_slc_${slc}" \
  --latency_ref 3190 \
  vggt
```

### VGGT Result
```
[PointCloud Summary]
  Samples     : 5
  Latency (ms): 311.28
  Speedup     : 10.25x (ref=3190.00 ms)
  Completeness: 0.0000
  Accuracy    : 0.0000
```

### Co-Me Command
```
seq=1
slc=5

python -m Eval.vggt_cloud_point \
  --dataset eth-3d \
  --seq_l "$seq" \
  --slice "$slc" \
  --result "ours_results/eth3d_ours_pointcloud_seq_${seq}_slc_${slc}.json" \
  --masks "ours_results/eth3d_ours_masks_seq_${seq}_slc_${slc}" \
  --latency_ref 3190 \
  ours \
  --grp_size 4 \
  --mask_setup bot-p 0.5 \
  --accelerator ./Model/VGGT_Accelerator/checkpoint.pth
```

### Co-Me Result
```
[PointCloud Summary]
  Samples     : 5
  Latency (ms): 308.34
  Speedup     : 10.35x (ref=3190.00 ms)
  Completeness: 11.8446
  Accuracy    : 8.3264
```

---

## Frame 4 / Slice 5

### VGGT Command
```
seq=4
slc=5

python -m Eval.vggt_cloud_point \
  --dataset eth-3d \
  --seq_l "$seq" \
  --slice "$slc" \
  --result "ours_results/eth3d_vggt_pointcloud_seq_${seq}_slc_${slc}.json" \
  --masks "ours_results/eth3d_vggt_masks_seq_${seq}_slc_${slc}" \
  --latency_ref 3190 \
  vggt
```

### VGGT Result
```
(pending)
```

### Co-Me Command
```
seq=4
slc=5

python -m Eval.vggt_cloud_point \
  --dataset eth-3d \
  --seq_l "$seq" \
  --slice "$slc" \
  --result "ours_results/eth3d_ours_pointcloud_seq_${seq}_slc_${slc}.json" \
  --masks "ours_results/eth3d_ours_masks_seq_${seq}_slc_${slc}" \
  --latency_ref 3190 \
  ours \
  --grp_size 4 \
  --mask_setup bot-p 0.5 \
  --accelerator ./Model/VGGT_Accelerator/checkpoint.pth
```

### Co-Me Result
```
(pending)
```

---

## Frame 8 / Slice 1

### VGGT Command
```
seq=8
slc=1

python -m Eval.vggt_cloud_point \
  --dataset eth-3d \
  --seq_l "$seq" \
  --slice "$slc" \
  --result "ours_results/eth3d_vggt_pointcloud_seq_${seq}_slc_${slc}.json" \
  --masks "ours_results/eth3d_vggt_masks_seq_${seq}_slc_${slc}" \
  --latency_ref 3190 \
  vggt
```

### VGGT Result
```
(pending)
```

### Co-Me Command
```
seq=8
slc=1

python -m Eval.vggt_cloud_point \
  --dataset eth-3d \
  --seq_l "$seq" \
  --slice "$slc" \
  --result "ours_results/eth3d_ours_pointcloud_seq_${seq}_slc_${slc}.json" \
  --masks "ours_results/eth3d_ours_masks_seq_${seq}_slc_${slc}" \
  --latency_ref 3190 \
  ours \
  --grp_size 4 \
  --mask_setup bot-p 0.5 \
  --accelerator ./Model/VGGT_Accelerator/checkpoint.pth
```

### Co-Me Result
```
(pending)
```

---

## Depth Evaluation (Reference)

These commands reproduce the depth-only runs (useful for source control snapshots).

### VGGT (Frame 16 / Slice 5)
```
python -m Eval.vggt_depth \
  --dataset eth-3d \
  --seq_l 16 \
  --slice 5 \
  --result ours_results/eth3d_vggt_depth_seq16_slice5.json \
  --masks ours_results/eth3d_vggt_depth_masks_seq16_slice5 \
  vggt
```

### Co-Me (Frame 8)
```
python -m Eval.vggt_depth \
  --dataset eth-3d \
  --seq_l 8 \
  --result ours_results/eth3d_ours_depth_seq8.json \
  --masks ours_results/eth3d_ours_depth_masks_seq8 \
  ours \
  --grp_size 4 \
  --mask_setup bot-p 0.5 \
  --accelerator ./Model/VGGT_Accelerator/checkpoint.pth
```
