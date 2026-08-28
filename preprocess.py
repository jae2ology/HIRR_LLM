# change the HOMER dataset into activity groups and append END token

import os
import json

def preprocess_homer_dataset(input_path, use_coarse=True):
    """ Passes HOMER households into sequence tuples:

        (activity, (obj_1, obj_2, ...., obj_n)

        raw data keys-------
        activity: string
        times: timestamps
        active_edges: relationship between user and objects (i.e, holds)
        nodes: dictionary mapping objects to their text label (i.e, 102: "bowl")
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # select activity level
    act_key = 'activities_coarse' if use_coarse and 'activities_coarse' in data else 'activities'
    activities_list = data.get(act_key, [])
    graphs = data.get('graphs', [])

    extracted_segments = []
    current_activity_name = None
    current_raw_sequence = []

    num_steps = min(len(activities_list), len(graphs))

    IGNORE_CATEGORIES = {'rooms', 'doors', 'walls', 'floor', 'ceiling', 'building', 'character'}
    IGNORE_CLASSES = {'character', 'person', 'agent', 'floor', 'wall', 'ceiling', 'room'}

    prev_edges = set()

    for t in range(num_steps):
        step_act = activities_list[t]
        step_act_str = str(step_act).strip() if step_act is not None else "idle"

        # skip idle state 1
        if step_act_str in ('1', 'idle', 'none', 'Idle', 'None', '', 'idle_state'):
            if current_activity_name is not None and len(current_raw_sequence) > 0:
                _finalize_segment(input_path, current_activity_name, current_raw_sequence, extracted_segments)
                current_activity_name = None
                current_raw_sequence = []
            prev_edges = set()
            continue

        # split segment when activity changes
        if current_activity_name is not None and step_act_str != current_activity_name:
            _finalize_segment(input_path, current_activity_name, current_raw_sequence, extracted_segments)
            current_raw_sequence = []
            prev_edges = set()

        current_activity_name = step_act_str
        graph_t = graphs[t]

        step_objects = []

        if isinstance(graph_t, dict):
            nodes = graph_t.get('nodes', [])
            edges = graph_t.get('edges', [])

            id_to_node = {n['id']: n for n in nodes if 'id' in n}

            # Represent current step edges as set of tuples: (from_id, relation, to_id)
            curr_edges = set()
            for edge in edges:
                f_id = edge.get('from_id', edge.get('id_src'))
                t_id = edge.get('to_id', edge.get('id_tar'))
                rel = str(edge.get('relation', edge.get('relation_type', ''))).upper()
                curr_edges.add((f_id, rel, t_id))

            # Detect state-changed edges (edges added or changed since previous step)
            if prev_edges:
                changed_edges = curr_edges - prev_edges
            else:
                changed_edges = set()

            prev_edges = curr_edges

            # Collect objects involved in relation changes
            for f_id, rel, t_id in changed_edges:
                for node_id in (f_id, t_id):
                    if node_id in id_to_node:
                        node = id_to_node[node_id]
                        cat = str(node.get('category', '')).lower()
                        class_name = str(node.get('class_name', '')).lower()

                        if cat not in IGNORE_CATEGORIES and class_name not in IGNORE_CLASSES:
                            if class_name not in step_objects:
                                step_objects.append(class_name)

        # append step's unique objects to sequence
        for obj in step_objects:
            current_raw_sequence.append(obj)

    # catch remaining segment at end of file
    if current_activity_name is not None and len(current_raw_sequence) > 0:
        _finalize_segment(input_path, current_activity_name, current_raw_sequence, extracted_segments)

    return extracted_segments


def _finalize_segment(file_path, act_id, raw_seq, output_list):
    # get rid of consecutive identical object interactions
    cleaned_sequence = []
    for obj in raw_seq:
        if not cleaned_sequence or cleaned_sequence[-1] != obj:
            cleaned_sequence.append(obj)

    # append END token
    if len(cleaned_sequence) >= 1:
        cleaned_sequence.append("END")
        output_list.append({
            "file": os.path.basename(file_path),
            "activity": str(act_id),
            "sequence": cleaned_sequence
        })


def process_directory(dir_path, output_json):
    results = []

    if os.path.exists(dir_path):
        files = sorted([f for f in os.listdir(dir_path) if f.endswith('.json')])
        for file_name in files:
            file_path = os.path.join(dir_path, file_name)
            file_segments = preprocess_homer_dataset(file_path)
            if file_segments:
                results.extend(file_segments)
                for res in file_segments:
                    print(f"[{file_name}] {res['activity']}: {res['sequence']}")

    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)
    print(f"\nSaved {len(results)} sequence tuples to {output_json}")


def merge_json_files(file_list, output_filename):
    merged_data = []

    for file_path in file_list:
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                merged_data.extend(data)
                print(f"Loaded {len(data)} items from {file_path}")
        else:
            print(f"Warning: File not found: {file_path}")

    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, indent=4)

    print(f"Successfully saved merged file ({len(merged_data)} total items) -> {output_filename}\n")
    return merged_data


# --- File Paths ---
# Adjust filenames/paths to match your generated household JSON outputs
train_files = [
    'homer_train_tuples_A.json',
    'homer_train_tuples_B.json',
    'homer_train_tuples_C.json'
]

test_files = [
    'homer_test_tuples_A.json',
    'homer_test_tuples_B.json',
    'homer_test_tuples_C.json'
]

# --- 1. Merge all Train files ---
print("--- Merging Training Datasets ---")
merged_train = merge_json_files(train_files, 'homer_train_all.json')

# --- 2. Merge all Test files ---
print("--- Merging Test Datasets ---")
merged_test = merge_json_files(test_files, 'homer_test_all.json')

# --- 3. (Optional) Merge EVERYTHING into a single Dataset file ---
print("--- Merging Complete Dataset ---")
complete_dataset = merged_train + merged_test
with open('homer_dataset_complete.json', 'w', encoding='utf-8') as f:
    json.dump(complete_dataset, f, indent=4)
print(f"Saved complete dataset ({len(complete_dataset)} total tuples) -> homer_dataset_complete.json")