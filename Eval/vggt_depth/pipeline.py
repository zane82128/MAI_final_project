import torch
from torchvision.io import read_image
from pathlib import Path
from Network.vggt import get_VGGT
from Accelerate.vggt import accelerate_vggt, Accelerator_Config

# 設定路徑和參數
image_path = "your_image.jpg"  # 修改為你的圖片路徑
output_dir = Path("./inference_outputs")
output_dir.mkdir(parents=True, exist_ok=True)

# 1) 讀取單張圖片
image = read_image(image_path).float() / 255.0  # (3, H, W)

# 2) 為了 VGGT 的序列輸入，重複該圖片（或準備序列）
# 假設序列長度為 S (根據你的需求調整)
S = 1  # 或改為你需要的序列長度
images = image.unsqueeze(0).repeat(S, 1, 1, 1).unsqueeze(0)  # (B=1, S, 3, H, W)

# 3) 根據模型輸入大小補/縮圖
infer_shape = (27 * 14, 36 * 14)  # 378 x 504
images = torch.nn.functional.interpolate(
    images.flatten(0, 1), size=infer_shape, mode='bilinear', align_corners=False
).view(1, S, 3, *infer_shape)

# 4) 建立模型並加載加速器
model = get_VGGT()
model, get_mask, set_mask = accelerate_vggt(
    model,
    Accelerator_Config(
        accelerator=Path("./Model/VGGT_Accelerator/checkpoint.pth"),
        grp_size=4,
        mask_setup=("bot-p", 0.5),
    )
)
model.cuda().eval()

# 5) 執行推論
with torch.no_grad():
    images = images.cuda()
    
    # 調用 aggregator 獲得 tokens
    aggregated_tokens_list, ps_idx = model.aggregator(images)
    
    # 調用 depth_head 獲得 depth 和 confidence
    assert model.depth_head is not None
    depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images, ps_idx)
    
    # depth_map: (B, S, C, H_orig, W_orig) or similar
    # depth_conf: confidence map from VGGT
    
    # 6) 獲取 CoMe 的 mask
    come_mask = get_mask()
    
    # 7) 保存結果
    torch.save({
        "depth": depth_map.cpu(),
        "vggt_confidence": depth_conf.cpu(),
        "come_mask": come_mask.cpu() if come_mask is not None else None,
    }, output_dir / "inference_result.pth")
    
    print(f"Inference complete! Results saved to {output_dir}")