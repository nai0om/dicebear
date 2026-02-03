"""
ByteDance SEED Tokenizer Thai Test

Test Thai tokenization with ByteDance's SEED models.

SEED 1.8 is ByteDance's flagship model (API via Volcano Engine).
Open-source alternatives: Seed-Coder-8B, Seed-OSS-36B

Requirements:
    pip install transformers torch

Usage:
    python test_seed_tokenizer.py

Note: Requires network access to HuggingFace to download the tokenizer.

References:
    - https://github.com/ByteDance-Seed/Seed-1.8
    - https://huggingface.co/ByteDance-Seed
    - https://seed.bytedance.com/en/seed1_8
"""
from transformers import AutoTokenizer

# ByteDance Seed model (open-source version)
# Options:
#   - "ByteDance-Seed/Seed-Coder-8B-Instruct" (8B, code-focused)
#   - "ByteDance-Seed/Seed-OSS-36B-Instruct" (36B, general)
model_id = "ByteDance-Seed/Seed-Coder-8B-Instruct"

print("=" * 50)
print("ByteDance SEED Tokenizer Thai Test")
print("=" * 50)
print(f"\nLoading tokenizer: {model_id}")

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

thai_text = "สวัสดีครับ"
print(f"\nThai text: {thai_text}")

# Tokenize
tokens = tokenizer.tokenize(thai_text)
ids = tokenizer.convert_tokens_to_ids(tokens)

print(f"\nTokens: {tokens}")
print(f"Token IDs: {ids}")
print(f"Number of tokens: {len(tokens)}")

# Also test encoding/decoding
encoded = tokenizer.encode(thai_text)
decoded = tokenizer.decode(encoded)
print(f"\nEncoded IDs: {encoded}")
print(f"Decoded text: {decoded}")

print("\n" + "=" * 50)
