import json
import pandas as pd


def process_sequence(seq):
    # remove duplicate items while keeping the order
    unique_seq = list(dict.fromkeys(seq))

    # check if the sequence ends with "END" and has at least two items
    if len(unique_seq) >= 2 and unique_seq[-1] == "END":
        # replace the item before "END" with the mask
        unique_seq[-2] = "_________"

    return unique_seq


def process_dataset(input_file_path, output_file_path):
    with open(input_file_path, 'r') as f:
        data = json.load(f)

    for item in data:
        if "sequence" in item:
            item["sequence"] = process_sequence(item["sequence"])

    with open(output_file_path, 'w') as f:
        json.dump(data, f, indent=4)


def process_train_dataset(input_path, output_path):
    with open(input_path, 'r') as f:
        data = json.load(f)

    cleaned_data = []

    for item in data:
        unique_sequence = list(dict.fromkeys(item["sequence"]))

        # delete the "file" key
        cleaned_item = {
            "activity": item["activity"],
            "sequence": unique_sequence
        }

        cleaned_data.append(cleaned_item)

    with open(output_path, 'w') as f:
        json.dump(cleaned_data, f, indent=4)


def load_activity_data(file_path):
    """extracts object features to predict the activity"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    X_rows, y_labels = [], []

    for item in data:
        seq = list(dict.fromkeys(item["sequence"]))
        if seq and seq[-1] == "END":
            seq = seq[:-1]

        # use recent objects as features for Naive Bayes
        last_obj = seq[-1] if len(seq) >= 1 else "<NONE>"
        prev_obj = seq[-2] if len(seq) >= 2 else "<NONE>"

        X_rows.append({"prev_object": prev_obj, "last_object": last_obj})
        y_labels.append(item["activity"])

    return pd.DataFrame(X_rows), pd.Series(y_labels)