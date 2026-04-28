# pip install transformers -q
from transformers import pipeline

# --- GENERATIVE AI: CREATING NEW CONTENT ---
# We load a small, fast Large Language Model (LLM) right here in the browser.
print("Loading GenAI Model (this takes a few seconds)...\n")
generator = pipeline('text-generation', model='distilgpt2')

prompt = "Our new startup in India will combine Agentic AI and linear optical quantum computing to"

print("=====================================================")
print(f"  USER PROMPT: '{prompt}'")
print("=====================================================\n")

# 1. Low Temperature (Predictable, safe text)
print("--- Generation 1: Strict & Safe (Low Temperature) ---")
safe_output = generator(prompt, max_new_tokens=20, do_sample=True, temperature=0.3, pad_token_id=50256)
print(safe_output[0]['generated_text'])
print("\nTakeaway: The model predicts the most mathematically logical next words.\n")

# 2. High Temperature (Creative, but risks Hallucination)
print("--- Generation 2: High Creativity (High Temperature) ---")
chaos_output = generator(prompt, max_new_tokens=30, do_sample=True, temperature=2.0, pad_token_id=50256)
print(chaos_output[0]['generated_text'])
print("\nTakeaway: By forcing the model to pick less likely words, we get 'creativity'. But if we push it too far, we get Hallucinations—it starts outputting wrong generated facts or complete nonsense.")
print("=====================================================")