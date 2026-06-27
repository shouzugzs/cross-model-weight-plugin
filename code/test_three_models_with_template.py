import torch
import re
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_DIR = r"D:\WhiteBox_MiniCPM"
MODEL_PATHS = {
    "Base": r"D:\WhiteBox_MiniCPM\MiniCPM5-1B-Base",
    "SFT": r"D:\WhiteBox_MiniCPM\MiniCPM5-1B-SFT",
    "RL": r"D:\WhiteBox_MiniCPM\MiniCPM5-1B",
}
USER_QUESTION = "请写一首关于冬天的五言绝句"

def generate_with_model(model_path, model_name, enable_thinking=False):
    print(f"\n{'='*60}\n加载模型: {model_name} (思维链: {enable_thinking})")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        trust_remote_code=True,
        device_map="cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # 关键：通过模型配置控制思维链
    if hasattr(model, 'config'):
        model.config.enable_thinking = enable_thinking
        print(f"  已设置 enable_thinking = {model.config.enable_thinking}")

    # 手动构造 chatml 格式的 prompt
    prompt = f"<|im_start|>user\n{USER_QUESTION}\n<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)

    gen_kwargs = {
        "max_new_tokens": 256,
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.9,
        "repetition_penalty": 1.05,
        "no_repeat_ngram_size": 3,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "early_stopping": False,
    }

    outputs = model.generate(**inputs, **gen_kwargs)
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=False)

    # 如果关闭了思维链，但输出仍有 <think> 块，手动清理
    if not enable_thinking:
        full_text = re.sub(r'<think>.*?</think>', '', full_text, flags=re.DOTALL)
        full_text = re.sub(r'</?think>', '', full_text)
        full_text = re.sub(r'\n\s*\n', '\n', full_text).strip()

    print(f"\n【{model_name} 完整输出】:\n{full_text}")
    del model, tokenizer
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

# ========== 测试关闭思维链 ==========
print("🔹 测试关闭思维链 (快速回答)")
for name, path in MODEL_PATHS.items():
    generate_with_model(path, name, enable_thinking=False)

# ========== 测试开启思维链（仅 SFT 和 RL，Base 可能不支持） ==========
print("\n\n🔹 测试开启思维链 (深度推理) —— 仅 SFT/RL")
for name, path in MODEL_PATHS.items():
    if name != "Base":  # Base 可能没有思维链功能
        generate_with_model(path, name, enable_thinking=True)