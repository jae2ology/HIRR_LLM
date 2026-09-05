import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import OrdinalEncoder

from models.activity_recognition import ActivityRecognizer
from data.merged_json.process_train_test import load_activity_data
from models.markov_model import SequenceModel


# CATEGORICAL NAIVE BAYES (F1, Confusion Matrix)
dataset_path = 'data/merged_json/dataset.json'
X_all_act, y_all_act = load_activity_data(dataset_path) # helper method that gets rid of the 'activity' and only keeps the objects for the model to predict with for X, and has the list of activities for Y

# encode features to turn into numbers
enc_act = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
X_all_act_enc = enc_act.fit_transform(X_all_act)

# handle unseen categories (-1)
X_all_act_enc[X_all_act_enc == -1] = X_all_act_enc.max() + 1

# --- F1 Score & Confusion Matrix ---
X_train_act, y_train_act = load_activity_data('data/merged_json/train/train.json')
X_test_act, y_test_act = load_activity_data('data/merged_json/test/test.json')

X_train_enc = enc_act.fit_transform(X_train_act)
X_test_enc = enc_act.transform(X_test_act)
X_test_enc[X_test_enc == -1] = X_train_enc.max() + 1

holdout_nb_model = ActivityRecognizer(X_train_enc, y_train_act)
nb_preds = holdout_nb_model.predict(X_test_enc)

nb_f1_macro = f1_score(y_test_act, nb_preds, average='macro')
nb_f1_weighted = f1_score(y_test_act, nb_preds, average='weighted')
print(f"Naive Bayes F1-Score:    {nb_f1_macro * 100:.2f}%")
print(f"Naive Bayes Weighted F1-Score: {nb_f1_weighted * 100:.2f}%")

# Plot Naive Bayes Confusion Matrix
plt.figure(figsize=(8, 6))
labels = np.unique(y_test_act)
cm_nb = confusion_matrix(y_test_act, nb_preds, labels=labels)
sns.heatmap(cm_nb, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title('Naive Bayes Activity Recognition - Confusion Matrix')
plt.xlabel('Predicted Activity')
plt.ylabel('True Activity')
plt.tight_layout()
plt.show()

# 2. MARKOV SEQUENCE MODEL (F1, Confusion Matrix)
# F1 Score & Confusion Matrix
with open('data/merged_json/test/test.json', 'r', encoding='utf-8') as f:
    test_json_data = json.load(f)

holdout_markov = SequenceModel(n=1, threshold=0.5)
holdout_markov.train('data/merged_json/train/train.json')

y_markov_true, y_markov_pred = [], []

for i, item in enumerate(test_json_data):
    pred_act = nb_preds[i]  # Using predicted activity from Naive Bayes
    seq = list(dict.fromkeys(item["sequence"]))
    if seq and seq[-1] == "END":
        seq = seq[:-1]

    if len(seq) >= 2:
        target = seq[-1]
        context = seq[:-1]
        top_pred, _, _, _ = holdout_markov.predict_next(pred_act, context)

        y_markov_true.append(target)
        y_markov_pred.append(top_pred if top_pred is not None else "<UNKNOWN>")

markov_f1_macro = f1_score(y_markov_true, y_markov_pred, average='macro', zero_division=0)
markov_f1_weighted = f1_score(y_markov_true, y_markov_pred, average='weighted', zero_division=0)

print(f"Markov Macro F1-Score:    {markov_f1_macro * 100:.2f}%")
print(f"Markov Weighted F1-Score: {markov_f1_weighted * 100:.2f}%")

# Plot Top-10 Most Common Objects Confusion Matrix for Markov Model
obj_labels = pd.Series(y_markov_true).value_counts().head(10).index.tolist()
cm_markov = confusion_matrix(y_markov_true, y_markov_pred, labels=obj_labels)

plt.figure(figsize=(10, 8))
sns.heatmap(cm_markov, annot=True, fmt='d', cmap='Greens', xticklabels=obj_labels, yticklabels=obj_labels)
plt.title('Markov Model Next-Object Prediction - Top 10 Objects Confusion Matrix')
plt.xlabel('Predicted Next Object')
plt.ylabel('True Next Object')
plt.tight_layout()
plt.show()