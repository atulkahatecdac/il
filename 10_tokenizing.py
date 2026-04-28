# --- INSTALLATION ---
# (Run this first if 'transformers' is not yet installed)
# !pip install transformers -q

from transformers import AutoTokenizer

print("=====================================================")
print("      LAB: HOW AI READS (TOKENIZATION EXPOSED)       ")
print("=====================================================\n")

print("Loading a production GenAI Tokenizer (GPT-2)...")
# We use the tokenizer from GPT-2, a classic foundational model
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# The test sentence
text = "Unbelievable! AI is transforming production workflows."

print("=====================================================")
print(f"  HUMAN INPUT: '{text}'")
print("=====================================================\n")

# --- STEP 1: CHOPPING INTO TOKENS ---
# The tokenizer splits the text into known "sub-word" chunks
tokens = tokenizer.tokenize(text)
print("--- Step 1: The 'Tokens' ---")
print(tokens)
print("\nLook closely: 'Unbelievable' wasn't recognized as one word. It got chopped into 'Un', 'bel', 'iev', 'able'. This is how AI handles words it has never seen before!\n")

# --- STEP 2: CONVERTING TO MATH (IDs) ---
# Neural networks can't read letters, only numbers. We map tokens to their dictionary IDs.
token_ids = tokenizer.encode(text)
print("--- Step 2: The Mathematical IDs ---")
print(token_ids)
print("\nThis dense array of integers is the ONLY thing the Deep Learning model actually 'sees'.\n")

# --- STEP 3: DECODING ---
# When the AI generates a response, it spits out numbers. We have to translate them back.
print("--- Step 3: Decoding Back to English ---")
decoded_text = tokenizer.decode(token_ids)
print(f"Decoded Output: '{decoded_text}'")

print("\n=====================================================")