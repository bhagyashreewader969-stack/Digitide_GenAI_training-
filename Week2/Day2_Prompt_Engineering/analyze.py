import os
import re

def read_file(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read().strip()
        return content
    except FileNotFoundError:
        print(f"⚠️ File not found: {path}")
        return ""

def calculate_metrics(text):
    words = text.split()
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    word_count = len(words)
    sentence_count = len(sentences)
    avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
    lexical_diversity = len(set(words)) / len(words) if words else 0
    return {
        "word_count": word_count,
        "sentence_count": sentence_count,
        "avg_sentence_length": avg_sentence_length,
        "lexical_diversity": lexical_diversity
    }

role_text = read_file("outputs/role_output.txt")
cot_text = read_file("outputs/chain_of_thought_output.txt")

if not role_text or not cot_text:
    print("⚠️ One or both prompt outputs are empty. Please run main.py first.")
    exit()

role_metrics = calculate_metrics(role_text)
cot_metrics = calculate_metrics(cot_text)

# Save metrics to file
os.makedirs("outputs", exist_ok=True)
with open("outputs/metrics_readable.txt", "w", encoding="utf-8") as f:
    f.write(f"=== Prompt Output Comparison ===\n\n")
    f.write(f"Role-Based Prompt:\n")
    f.write(f"- Word count: {role_metrics['word_count']}\n")
    f.write(f"- Sentence count: {role_metrics['sentence_count']}\n")
    f.write(f"- Average sentence length: {role_metrics['avg_sentence_length']:.1f} words\n")
    f.write(f"- Lexical diversity: {role_metrics['lexical_diversity']:.2f}\n\n")
    
    f.write(f"Chain-of-Thought Prompt:\n")
    f.write(f"- Word count: {cot_metrics['word_count']}\n")
    f.write(f"- Sentence count: {cot_metrics['sentence_count']}\n")
    f.write(f"- Average sentence length: {cot_metrics['avg_sentence_length']:.1f} words\n")
    f.write(f"- Lexical diversity: {cot_metrics['lexical_diversity']:.2f}\n\n")

    f.write("Differences:\n")
    f.write(f"- Chain-of-Thought output is longer ({cot_metrics['word_count'] - role_metrics['word_count']} more words).\n")

# Print comparison directly in terminal
print("=== Prompt Output Comparison ===")
print("\nRole-Based Prompt:")
print(f"- Word count: {role_metrics['word_count']}")
print(f"- Sentence count: {role_metrics['sentence_count']}")
print(f"- Average sentence length: {role_metrics['avg_sentence_length']:.1f} words")
print(f"- Lexical diversity: {role_metrics['lexical_diversity']:.2f}")

print("\nChain-of-Thought Prompt:")
print(f"- Word count: {cot_metrics['word_count']}")
print(f"- Sentence count: {cot_metrics['sentence_count']}")
print(f"- Average sentence length: {cot_metrics['avg_sentence_length']:.1f} words")
print(f"- Lexical diversity: {cot_metrics['lexical_diversity']:.2f}")

print("\nDifferences:")
print(f"- Chain-of-Thought output is longer ({cot_metrics['word_count'] - role_metrics['word_count']} more words).")
