# --- INSTALLATION ---
# (Run this first in Colab if transformers is not installed)
# !pip install transformers -q

from transformers import pipeline

print("=====================================================")
print("      HUB 1: HUGGING FACE (NLP MODEL DOWNLOAD)       ")
print("=====================================================\n")

print("Downloading pre-trained model from Hugging Face Hub...")
# The 'pipeline' automatically finds, downloads, and loads the model weights
analyzer = pipeline("sentiment-analysis")
print("Model successfully downloaded and loaded into memory!\n")

# --- LIVE INFERENCE ---
# Testing the model with a complex, domain-specific input
test_text = "We are successfully stabilizing the qubits in our new linear optical quantum computer!"

print(f"User Input: '{test_text}'")
prediction = analyzer(test_text)

print("\n--- MODEL OUTPUT ---")
print(f"Classification: {prediction[0]['label']}")
print(f"Confidence Score: {round(prediction[0]['score'] * 100, 2)}%")
print("=====================================================")