import os
import torch

TEMP_DIR = r"D:\WhiteBox_MiniCPM\temp"
PLUGIN_DIR = r"D:\WhiteBox_MiniCPM\plugins"

print("📂 加载缓存的 Base 和 RL 权重...")
base_mlp = torch.load(os.path.join(TEMP_DIR, "base_mlp_weights.pt"), map_location="cpu")
rl_mlp = torch.load(os.path.join(TEMP_DIR, "rl_mlp_weights.pt"), map_location="cpu")
base_attn = torch.load(os.path.join(TEMP_DIR, "base_attn_weights.pt"), map_location="cpu")
rl_attn = torch.load(os.path.join(TEMP_DIR, "rl_attn_weights.pt"), map_location="cpu")

print("🔧 构建纯净差值 (RL - Base)...")
def build_delta(high, low):
    delta = {}
    for key in high.keys():
        if key in low:
            delta[key] = (high[key].float() - low[key].float()).half()
    return delta

delta_mlp = build_delta(rl_mlp, base_mlp)
delta_attn = build_delta(rl_attn, base_attn)

# 合并为一个完整的插件（或者保存两个，便于分开测试）
merged_delta = {**delta_mlp, **delta_attn}
torch.save(merged_delta, os.path.join(PLUGIN_DIR, "plugin_rl_minus_base_dense.pt"))

print(f"💾 纯净插件已保存: plugin_rl_minus_base_dense.pt")
print(f"   MLP 部分: {sum(v.numel() for v in delta_mlp.values()) / 1e6:.1f}M 参数")
print(f"   Attn 部分: {sum(v.numel() for v in delta_attn.values()) / 1e6:.1f}M 参数")