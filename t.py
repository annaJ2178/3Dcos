
import torch

# Load the .pt file

def find_lora_tensors(file_path):
    """Find and inspect LoRA-related tensors"""
    data = torch.load(file_path, map_location='cpu')
    
    print("=== LoRA Tensors ===")
    lora_tensors = {}
    
    if isinstance(data, dict):
        for key, value in data.items():

            if "net.blocks.14.self_attn.k_norm" in key: breakpoint()
            if 'lora' in key.lower() and torch.is_tensor(value):
                lora_tensors[key] = value
                print(f"\n{key}:")
                print(f"  Shape: {value.shape}")
                print(f"  Min: {value.min().item():.6f}")
                print(f"  Max: {value.max().item():.6f}")
                print(f"  Mean: {value.mean().item():.6f}")
                
                # Show snippet of values
                if value.numel() > 0:
                    flat_vals = value.flatten()
                    print(f"  Sample values: {flat_vals[:5].tolist()}")
    
    print(f"\nFound {len(lora_tensors)} LoRA tensors")
    return lora_tensors

# Usage
lora_tensors = find_lora_tensors('/nas/jiangyuxin/code/cosmos-predict2/checkpoints/cosmos_predict_v2p5/video2world/2b_calvin_debug/checkpoints/iter_000005000/model.pt')

# 'net.blocks.14.self_attn.k_norm.weight'
# (Pdb) value[:5]
# tensor([1.2500, 1.2344, 0.9844, 1.3047, 1.2109], dtype=torch.bfloat16)
# (Pdb) 



lora_tensors = find_lora_tensors('/nas/jiangyuxin/code/cosmos-predict2/checkpoints/posttraining/video2world_lora/2b_video2world_debug_fps2/checkpoints/model/iter_000010000.pt')
# (Pdb) value[:5]
# tensor([1.1172, 1.1250, 0.9062, 1.1719, 1.1250], dtype=torch.bfloat16)

lora_tensors = find_lora_tensors("/nas/jiangyuxin/huggingface/cache/models--nvidia--Cosmos-Predict2-2B-Video2World/snapshots/f50c09f5d8ab133a90cac3f4886a6471e9ba3f18/model-720p-16fps.pt")
# > /nas/jiangyuxin/code/cosmos-predict2.5/t.py(17)find_lora_tensors()
# -> if 'lora' in key.lower() and torch.is_tensor(value):
# (Pdb) key
# 'net.blocks.14.self_attn.k_norm.weight'
# (Pdb) value[:5]
# tensor([1.1172, 1.1250, 0.9141, 1.1875, 1.1406], dtype=torch.bfloat16)

lora_tensors = find_lora_tensors('/nas/jiangyuxin/huggingface/cache/models--nvidia--Cosmos-Predict2.5-2B/snapshots/1a7f55340992562b20e81a93238e6722345c855d/base/post-trained/81edfebe-bd6a-4039-8c1d-737df1a790bf_ema_bf16.pt')

# > /nas/jiangyuxin/code/cosmos-predict2.5/t.py(17)find_lora_tensors()
# -> if 'lora' in key.lower() and torch.is_tensor(value):
# (Pdb) value[:5]
# tensor([1.2656, 1.2500, 0.9922, 1.3203, 1.2188], dtype=torch.bfloat16)
# (Pdb) key
# 'net.blocks.14.self_attn.k_norm.weight'