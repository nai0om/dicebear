"""
Thai Tokenization Test

This script tests Thai tokenization using multiple methods:
1. PyThaiNLP (local library for Thai NLP)
2. Transformers AutoTokenizer (requires network access to HuggingFace)
"""

from pythainlp import word_tokenize

# Test Thai text
thai_text = "สวัสดีครับ"

print("=" * 50)
print("Thai Tokenization Test")
print("=" * 50)
print(f"\nInput Thai text: {thai_text}")

# Method 1: PyThaiNLP word tokenization (various engines)
print("\n--- PyThaiNLP Tokenization ---")

# Default engine (newmm - dictionary-based)
tokens_newmm = word_tokenize(thai_text, engine="newmm")
print(f"newmm engine: {tokens_newmm}")

# Longest matching
tokens_longest = word_tokenize(thai_text, engine="longest")
print(f"longest engine: {tokens_longest}")

# Try Transformers tokenizer (may fail due to network restrictions)
print("\n--- Transformers Tokenization (SEED) ---")
try:
    from transformers import AutoTokenizer
    model_id = "AILab-CVC/seed-tokenizer-2"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokens = tokenizer.tokenize(thai_text)
    ids = tokenizer.convert_tokens_to_ids(tokens)
    print(f"Tokens: {tokens}")
    print(f"Token IDs: {ids}")
except Exception as e:
    print(f"Could not load SEED tokenizer: {type(e).__name__}")
    print("(This is expected if there's no network access to HuggingFace)")

print("\n" + "=" * 50)
