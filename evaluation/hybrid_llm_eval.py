import os
import sys
import json
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sklearn.metrics import accuracy_score, f1_score
from models.baseline_HIRR import HIRRBase, UserAction

train_path = '../data/merged_json/train/train.json'
test_path  = '../data/merged_json/test/test.json'

hirr_base = HIRRBase(train_path)

with open(test_path, 'r', encoding='utf-8') as f:
    test_data = json.load(f)

y_true = []
y_pred_hybrid_llm = []

llm_calls_made = 0
total_evaluations = 0

for item in test_data:
    activity = item["activity"]
    seq = list(dict.fromkeys(item["sequence"]))
    if seq and seq[-1] == "END":
        seq = seq[:-1]

    if len(seq) >= 2:
        target = seq[-1]
        context = seq[:-1]

        action = UserAction(activity=activity, objects_in_activity=context, completed=False)
        top_pred = hirr_base.evaluate_next_step(action)

        y_true.append(target)
        y_pred_hybrid_llm.append(top_pred)
        total_evaluations += 1

# F1 SCORES
hybrid_f1_macro = f1_score(y_true, y_pred_hybrid_llm, average='macro', zero_division=0)
hybrid_f1_weighted = f1_score(y_true, y_pred_hybrid_llm, average='weighted', zero_division=0)

print("=" * 60)
print(" EVALUATION RESULTS WITH LLM FALLBACK")
print("=" * 60)
print(f"Total Test Instances:         {total_evaluations}")
print("-" * 60)
print(f"Hybrid + LLM F1-Score (Weighted): {hybrid_f1_weighted * 100:.2f}%")
print(f"Hybrid + LLM F1-Score (Macro):    {hybrid_f1_macro * 100:.2f}%")
print("=" * 60)