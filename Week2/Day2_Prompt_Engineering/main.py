from transformers import pipeline

# Load model
generator = pipeline("text2text-generation", model="google/flan-t5-large", device=-1)

# Role-based prompt
role_prompt = "You are a high school biology teacher. Explain photosynthesis to students in simple words."
role_output = generator(role_prompt, max_length=300)[0]['generated_text']

# Chain-of-thought prompt
cot_prompt = "Explain photosynthesis step by step, reasoning each step clearly."
cot_output = generator(cot_prompt, max_length=300)[0]['generated_text']

# Save outputs
os.makedirs("outputs", exist_ok=True)
with open("outputs/role_output.txt", "w", encoding="utf-8") as f:
    f.write(role_output)
with open("outputs/chain_of_thought_output.txt", "w", encoding="utf-8") as f:
    f.write(cot_output)

print("Role-based output saved to outputs/role_output.txt")
print("Chain-of-thought output saved to outputs/chain_of_thought_output.txt")
