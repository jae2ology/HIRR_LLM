import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import OrdinalEncoder
from models.activity_recognition import ActivityRecognizer
from data.merged_json.process_train_test import load_activity_data
from models.markov_model import SequenceModel

train_path = 'data/merged_json/train/train.json'
test_path = 'data/merged_json/test/test.json'

# load the activity data for the NB model
X_train_act, y_train_act = load_activity_data('data/merged_json/train/train.json')
X_test_act, y_test_act   = load_activity_data('data/merged_json/test/test.json')

# encode categorical features into numbers
enc_act = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
X_train_act_enc = enc_act.fit_transform(X_train_act)
X_test_act_enc  = enc_act.transform(X_test_act)

# replace unknown test values (-1) with a new nunber
X_test_act_enc[X_test_act_enc == -1] = X_train_act_enc.max() + 1

# setup the model using ENCODED data
activity_model = ActivityRecognizer(X_train_act_enc, y_train_act)

# test the categorical NB model using ENCODED data ->>>>>>>>>>>
act_preds = activity_model.predict(X_test_act_enc)
print(f"Categorical NB accuracy: {accuracy_score(y_test_act, act_preds) * 100:.2f}%")
print("\nClassification Report (Categorical NB):")
print(classification_report(y_test_act, act_preds))

# confusion matrix
labels = np.unique(y_test_act)
cm = confusion_matrix(y_test_act, act_preds, labels=labels)

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels, yticklabels=labels)
plt.title('Activity Recognition (Categorical NB) Confusion Matrix')
plt.xlabel('Predicted Activity')
plt.ylabel('Actual Activity')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


# markov model ------------------->
# train the markov model with the train set
markov = SequenceModel(n=1, threshold=0.5)
markov.train(train_path)

with open(test_path, 'r', encoding='utf-8') as f:
    test_data = json.load(f)

# test the markov model
y_true_items = []
top_1_e2e, top_3_e2e, top_5_e2e = [], [], []
llm_triggered_count = 0
llm_triggered_correct_count = 0
total_samples = 0

for i, item in enumerate(test_data):
    seq = list(dict.fromkeys(item["sequence"]))
    if seq and seq[-1] == "END":
        seq = seq[:-1]

    if len(seq) >= 2:
        target = seq[-1]
        context = seq[:-1]
        pred_activity = act_preds[i]  # predicted activity

        top_pred, max_prob, candidates, trigger_llm = markov.predict_next(pred_activity, context)

        # extract top-K candidate names
        candidate_names = [cand[0] for cand in candidates]

        # evaluate top 1 hit
        is_top1 = (top_pred == target)
        top_1_e2e.append(is_top1)

        # evaluate top-3 hit
        is_top3 = (target in candidate_names[:3])
        top_3_e2e.append(is_top3)

        # evaluate top-5 hit
        is_top5 = (target in candidate_names[:5])
        top_5_e2e.append(is_top5)

        # LLM trigger counts
        total_samples += 1
        if trigger_llm:
            llm_triggered_count += 1
            if is_top1:
                llm_triggered_correct_count += 1

top1_acc = (sum(top_1_e2e) / total_samples) * 100
top3_acc = (sum(top_3_e2e) / total_samples) * 100
top5_acc = (sum(top_5_e2e) / total_samples) * 100

trigger_rate = (llm_triggered_count / total_samples) * 100

print(f"Top-1 Accuracy: {top1_acc:.2f}%")
print(f"Top-3 Accuracy: {top3_acc:.2f}%")
print(f"Top-5 Accuracy: {top5_acc:.2f}%")
print("-" * 40)
print(f"Total Test Sequences Evaluated: {total_samples}")
print(f"LLM Fallback Triggered:        {llm_triggered_count} times ({trigger_rate:.2f}%)")

