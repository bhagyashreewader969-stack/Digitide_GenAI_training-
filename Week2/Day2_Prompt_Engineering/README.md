# Zero-Shot vs Few-Shot Accuracy Comparison (google/flan-t5-large)

This project runs both **zero-shot** and **few-shot** sentiment classification using Hugging Face's `google/flan-t5-large` model, 
and compares their accuracy against a ground truth label set.

## Files
- `compare.py` — Runs both models, collects predictions, and prints an accuracy report.
- `requirements.txt` — Dependencies list.
- `prompt_few.txt` — Few-shot examples for the few-shot model.
- `test_data.txt` — Sentences with ground truth labels in `sentence | label` format.

## How to Run
```bash
pip install -r requirements.txt

python compare.py
```

The script will output:
- Predictions from zero-shot and few-shot
- Accuracy for each
- Side-by-side comparison
