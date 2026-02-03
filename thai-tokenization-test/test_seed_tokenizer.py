"""
SEED Tokenizer Thai Test

Test Thai tokenization with the SEED tokenizer from AILab-CVC.

Requirements:
    pip install transformers torch

Usage:
    python test_seed_tokenizer.py

Note: Requires network access to HuggingFace to download the tokenizer.
"""
from transformers import AutoTokenizer

# SEED tokenizer model ID
model_id = "AILab-CVC/seed-tokenizer-2"

print("Loading SEED tokenizer...")
print(f"Model: {model_id}")
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

thai_text = "สวัสดีครับ"
print(f"\nThai text: {thai_text}")

tokens = tokenizer.tokenize(thai_text)
ids = tokenizer.convert_tokens_to_ids(tokens)

print(f"Tokens: {tokens}")
print(f"Token IDs: {ids}")
