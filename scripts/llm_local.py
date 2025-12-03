# scripts/llm_local.py
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Fully open, small chat model (no token needed once downloaded)
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

print(f"Loading local model: {MODEL_NAME} ...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
)

# Text-generation pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    temperature=0.1,
    do_sample=False,
)


def generate_completion(prompt: str) -> str:
    """
    Run the local model and return ONLY the new text it generates.
    """
    # For many chat models, pipeline returns the full text including prompt,
    # so we strip the prompt prefix if present.
    out = pipe(prompt)[0]["generated_text"]
    if out.startswith(prompt):
        out = out[len(prompt):]
    return out.strip()
