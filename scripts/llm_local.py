# scripts/llm_local.py
import os
import sys

# Global lazy-loaded components
_tokenizer = None
_model = None
_device = None
_api_client = None


def _load_local_model():
    """Lazily load the local model only when needed."""
    global _tokenizer, _model, _device
    
    if _model is not None:
        return

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    # Default to Qwen, but allow override (e.g., "meta-llama/Meta-Llama-3-8B-Instruct")
    MODEL_NAME = os.getenv("HF_MODEL_NAME", "Qwen/Qwen1.5-7B-Chat")
    print(f"[LLM] Loading local model: {MODEL_NAME} ...")

    if torch.cuda.is_available():
        _device = torch.device("cuda")
        device_str = "cuda"
    elif torch.backends.mps.is_available():
        _device = torch.device("mps")
        device_str = "mps"
    else:
        _device = torch.device("cpu")
        device_str = "cpu"
    
    print(f"[LLM] Using device: {device_str}")

    _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    dtype = (
        torch.bfloat16 if device_str == "cuda"
        else torch.float16 if device_str == "mps"
        else torch.float32
    )

    _model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=dtype,
    )
    _model.to(_device)
    _model.eval()


def generate_completion_local(prompt: str) -> str:
    import torch
    
    _load_local_model()
    
    messages = [
        {
            "role": "system",
            "content": (
                "You are a reproducibility assistant. "
                "You MUST answer by returning ONLY a JSON object, "
                "with no explanations or extra text."
            ),
        },
        {"role": "user", "content": prompt},
    ]

    chat_text = _tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = _tokenizer(chat_text, return_tensors="pt").to(_device)

    with torch.no_grad():
        outputs = _model.generate(
            **inputs,
            max_new_tokens=8192,
            do_sample=False,
        )

    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    out = _tokenizer.decode(generated_ids, skip_special_tokens=True)
    return out.strip()


def generate_completion_boisestate(prompt: str) -> str:
    """
    Call the boisestate.ai API using the user-provided request format.
    """
    import requests
    import json

    url = os.getenv("BOISESTATE_BASE_URL", "https://api.boisestate.ai/chat/api-converse")
    api_key = os.getenv("BOISESTATE_API_KEY")
    model_id = os.getenv("BOISESTATE_MODEL_ID", "openai.gpt-oss-120b-1:0")

    if not api_key:
        print("[ERROR] BOISESTATE_API_KEY not set.")
        return ""

    headers = {
        'X-API-Key': api_key,
        'Content-Type': 'application/json'
    }

    # The user provided prompt is usually the whole context, but this API expects a 'message'.
    # We'll pass the prompt as the message. 
    # NOTE: The system prompt from the caller might be lost if we just pass 'prompt'.
    # However, 'plan_with_llm.py' constructs a big string including the system instructions.
    # Let's check how build_prompt works. It returns a single string. 
    # So passing that single string as 'message' is the correct approach.

    payload = {
        'message': prompt,
        'modelId': model_id,
        'temperature': 0.1,  # Lower temperature for deterministic plans
        'maxTokens': 8192    # Ensure enough tokens for the JSON plan
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        
        # The user example shows data['text'] contains the response
        return data.get('text', '').strip()
        
    except Exception as e:
        print(f"[ERROR] Boisestate API call failed: {e}")
        # If response exists, print it for debugging
        if 'response' in locals():
            try:
                print(f"[DEBUG] Response: {response.text}")
            except:
                pass
        return ""


def generate_completion(prompt: str) -> str:
    """
    Dispatcher: uses boisestate.ai API if LLM_PROVIDER is 'boisestate',
    otherwise falls back to local Qwen model.
    """
    provider = os.getenv("LLM_PROVIDER", "local").lower()
    
    if provider == "boisestate":
        return generate_completion_boisestate(prompt)
    else:
        return generate_completion_local(prompt)

