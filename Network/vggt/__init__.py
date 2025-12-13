import torch
from pathlib import Path
from .models.vggt import VGGT


def get_VGGT():
    model = VGGT().cuda()  # ← 先移到 GPU，再加载权重
    
    cache_paths = [
        Path("/workspace/Model/local_cache/model.pt"),
        Path.home() / ".cache" / "torch" / "hub" / "checkpoints" / "model.pt",
        Path("/home/ubuntu/.cache/torch/hub/checkpoints/model.pt"),
    ]
    
    for cache_path in cache_paths:
        if cache_path.exists():
            print(f"loading local model: {cache_path}")
            model.load_state_dict(torch.load(cache_path, map_location='cuda'))
            return model
    
    print("local model doesnot exist, start download from HuggingFace...")
    URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    model.load_state_dict(torch.hub.load_state_dict_from_url(URL, map_location='cuda'))
    return model

def get_VGGT_vanilla():
    model = VGGT(vanilla=True).cuda()  # ← 先移到 GPU，再加载权重
    
    cache_paths = [
        Path("/workspace/Model/local_cache/model.pt"),
        Path.home() / ".cache" / "torch" / "hub" / "checkpoints" / "model.pt",
        Path("/home/ubuntu/.cache/torch/hub/checkpoints/model.pt"),
    ]
    
    for cache_path in cache_paths:
        if cache_path.exists():
            print(f"loading local model: {cache_path}")
            model.load_state_dict(torch.load(cache_path, map_location='cuda'))
            return model
    
    print("local model doesnot exist, start download from HuggingFace...")
    URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    model.load_state_dict(torch.hub.load_state_dict_from_url(URL, map_location='cuda'))
    return model
