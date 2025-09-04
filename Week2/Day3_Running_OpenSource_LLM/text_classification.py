from transformers import pipeline

# Load a small open-source sentiment-analysis model
classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# Test it
text = "I love using Hugging Face models!"
result = classifier(text)

print("Input:", text)
print("Prediction:", result)
