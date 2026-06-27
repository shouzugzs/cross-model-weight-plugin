import os
import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer

PATH_BASE = r"D:\WhiteBox_MiniCPM\MiniCPM5-1B-Base"
PLUGIN_ATTN = r"D:\WhiteBox_MiniCPM\plugins\plugin_sft_attn_dense.pt"

print("📂 1. 加载 Base 模型（纯净的诗人大脑）...")
model = AutoModelForCausalLM.from_pretrained(
    PATH_BASE,
    torch_dtype=torch.float32,
    trust_remote_code=True,
    device_map="cpu"
)
tokenizer = AutoTokenizer.from_pretrained(PATH_BASE, trust_remote_code=True)

print("📂 2. 加载 SFT Attention 插件（指令格式包）...")
plugin = torch.load(PLUGIN_ATTN, map_location="cpu")
print(f"   插件包含 {len(plugin)} 个层的增量")

# ---------- 核心操作：只注入 Attention，且只给 30% 的力度 ----------
print("🔥 3. 温柔注入 (仅 Attention 层, 缩放系数 0.3)...")
injected_count = 0
for name, param in model.named_parameters():
    if name in plugin:
        # 关键点：乘以 0.3，防止“软覆盖”
        delta = plugin[name].float() * 0.3
        param.data += delta
        injected_count += 1
print(f"   ✅ 已注入 {injected_count} 个 Attention 层")

# 释放插件内存
del plugin
gc.collect()

# ---------- 4. 生成测试（看它能否写出格式正确的诗） ----------
print("\n📝 4. 生成测试（请观察是否同时保留了文学性 + 格式感）...")
inputs = tokenizer("请写一首关于冬天的五言绝句", return_tensors="pt")
outputs = model.generate(
    **inputs,
    max_new_tokens=100,
    do_sample=True,
    temperature=0.85,
    repetition_penalty=1.2,
    no_repeat_ngram_size=3,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.eos_token_id,
)

print("\n【温柔注入后的输出】:")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))