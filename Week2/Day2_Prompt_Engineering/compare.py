from transformers import pipeline

def load_data(file_path):
    sentences, labels = [], []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if "|" in line:
                sent, label = line.strip().split("|")
                sentences.append(sent.strip())
                labels.append(label.strip())
    return sentences, labels

def build_few_shot_prompt(template, sentence):
    return template.format(sentence=sentence)

def predict_zero_shot(clf, sentences):
    preds = []
    for s in sentences:
        prompt = f"Determine if the following sentence is Positive or Negative: '{s}'"
        out = clf(prompt, max_new_tokens=5)[0]["generated_text"].strip()
        normalized = out.split()[0].strip().strip(".").capitalize()
        if normalized not in {"Positive", "Negative"}:
            normalized = out
        preds.append(normalized)
    return preds

def predict_few_shot(clf, template, sentences):
    preds = []
    for s in sentences:
        prompt = build_few_shot_prompt(template, s)
        out = clf(prompt, max_new_tokens=5)[0]["generated_text"].strip()
        normalized = out.split()[0].strip().strip(".").capitalize()
        if normalized not in {"Positive", "Negative"}:
            normalized = out
        preds.append(normalized)
    return preds

def accuracy(preds, labels):
    correct = sum(p == l for p, l in zip(preds, labels))
    return correct / len(labels) * 100

def main():
    sentences, labels = load_data("test_data.txt")
    model_id = "google/flan-t5-large"
    clf = pipeline("text2text-generation", model=model_id)

    # Load few-shot prompt
    with open("prompt_few.txt", "r", encoding="utf-8") as f:
        few_prompt = f.read()

    print("Running Zero-Shot Predictions...")
    zero_preds = predict_zero_shot(clf, sentences)
    print("Running Few-Shot Predictions...")
    few_preds = predict_few_shot(clf, few_prompt, sentences)

    print("\n--- Accuracy Report ---")
    print(f"Zero-Shot Accuracy: {accuracy(zero_preds, labels):.2f}%")
    print(f"Few-Shot Accuracy:  {accuracy(few_preds, labels):.2f}%\n")

    print("--- Side-by-Side ---")
    for i, (s, zl, fl, gt) in enumerate(zip(sentences, zero_preds, few_preds, labels), 1):
        print(f"{i}. {s}")
        print(f"   Zero-Shot: {zl} | Few-Shot: {fl} | Ground Truth: {gt}")

if __name__ == "__main__":
    main()
