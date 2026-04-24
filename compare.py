import json
import sys

def load(path):
    with open(path) as f:
        return json.load(f)

try:
    gpt4o = load("results_gpt4o.json")
    gpt35 = load("results_gpt35.json")
except FileNotFoundError as e:
    print(f"Missing results file: {e}")
    print("Run Eval.py and Eval_baseline.py first to generate results.")
    sys.exit(1)

def diff(a, b):
    d = a - b
    return f"+{d:.4f}" if d >= 0 else f"{d:.4f}"

print("\n" + "=" * 55)
print(f"  ROUGE Score Comparison — {gpt4o['num_samples']} samples each")
print("=" * 55)
print(f"{'Metric':<12} {'GPT-4o':>10} {'GPT-3.5-turbo':>14} {'Difference':>12}")
print("-" * 55)
for metric in ["rouge1", "rouge2", "rougeL"]:
    label = metric.upper().replace("ROUGE", "ROUGE-").replace("ROUGEL", "ROUGE-L")
    v4 = gpt4o[metric]
    v35 = gpt35[metric]
    print(f"{label:<12} {v4:>10.4f} {v35:>14.4f} {diff(v4, v35):>12}")
print("=" * 55)
print("Positive difference = GPT-4o scored higher\n")
