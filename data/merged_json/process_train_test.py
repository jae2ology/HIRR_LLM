import json


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


# Usage example:
process_train_dataset("data/merged_json/train/X_train_raw.json", "X_train.json")