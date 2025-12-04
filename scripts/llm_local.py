# scripts/llm_local.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen1.5-7B-Chat"

print(f"[LLM] Loading local model: {MODEL_NAME} ...")

# Device selection
if torch.cuda.is_available():
    device = torch.device("cuda")
    device_str = "cuda"
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    device_str = "mps"
else:
    device = torch.device("cpu")
    device_str = "cpu"

print(f"[LLM] Using device: {device_str}")

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

dtype = (
    torch.bfloat16 if device_str == "cuda"
    else torch.float16 if device_str == "mps"
    else torch.float32
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=dtype,
)
model.to(device)
model.eval()


def generate_completion(prompt: str) -> str:
    """
    Use Qwen chat format to generate a continuation for the given prompt.
    We wrap the prompt in a system+user conversation, then decode only
    the assistant's reply.
    """
    # Qwen is a chat model: we need messages + chat template
    messages = [
        {
            "role": "system",
            "content": (
                "You are a reproducibility assistant. "
                "You MUST answer by returning ONLY a JSON object, "
                "with no explanations or extra text."
            ),
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]

    # Build chat input text using model's chat template
    chat_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(chat_text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
        )

    # Strip the prompt part, keep only the newly generated tokens
    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    out = tokenizer.decode(generated_ids, skip_special_tokens=True)

    print("[LLM] Decoded output (first 200 chars):", repr(out[:200]))
    return out.strip()
