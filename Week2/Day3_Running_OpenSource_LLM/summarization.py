from transformers import pipeline

# Load summarization model
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

text = """Photosynthesis is the process by which green plants use sunlight 
to synthesize food from carbon dioxide and water. It involves the green pigment chlorophyll 
and generates oxygen as a byproduct."""

summary = summarizer(text, max_length=30, min_length=10, do_sample=False)

print("Summary:", summary[0]['summary_text'])
