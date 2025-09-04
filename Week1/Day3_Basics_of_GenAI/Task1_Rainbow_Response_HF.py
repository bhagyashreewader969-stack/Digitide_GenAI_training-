from transformers import pipeline, set_seed

# Create a text generation pipeline using a model like GPT-2
generator = pipeline('text-generation', model='gpt2')

# Set a seed for reproducibility (optional)
set_seed(42)

# Define the prompt
prompt = "Explain how rainbows are formed"

# Generate the response
responses = generator(prompt, max_length=100, num_return_sequences=1)

# Print the generated response
print("Response:")
print(responses[0]['generated_text'])
