from collections import defaultdict, Counter
import json


class SequenceModel:
    """
    N-gram sequence model built from scratch
    P(next_word | previous n words)
    """

    def __init__(self, n=1, threshold=0.5):
        self.n = n
        self.threshold = threshold
        # transitions[activity][context_tuple][next_object] = count
        self.transitions = defaultdict(lambda: defaultdict(Counter))

    def train(self, path_to_data: str):
        """Train the model by sliding an n-gram window over training sequences."""
        with open(path_to_data, 'r', encoding='utf-8') as f:
            dataset = json.load(f)

        for item in dataset:
            if isinstance(item, dict):
                activity = item.get("activity")
                sequence = item.get("sequence", [])
            else:
                activity, sequence = item[0], item[1]

            # keep the order and get rid of duplicate items & END token
            sequence = list(dict.fromkeys(sequence))
            if sequence and sequence[-1] == "END":
                sequence = sequence[:-1]

            # check to make sure theres at least (n + 1) items to form context + target
            if len(sequence) <= self.n:
                continue

            # here is the slide window: context of length self.n predicts sequence[i]
            for i in range(self.n, len(sequence)):
                context = tuple(sequence[i - self.n: i])
                next_obj = sequence[i]
                self.transitions[activity][context][next_obj] += 1

    def predict_next(self, activity, objects):
        # clean context items (remove end)
        objects = list(dict.fromkeys(objects))
        if objects and objects[-1] == "END":
            objects = objects[:-1]

        if len(objects) < self.n:
            return None, 0.0, [], True

        context = tuple(objects[-self.n:])

        if activity not in self.transitions or context not in self.transitions[activity]:
            return None, 0.0, [], True

        counts = self.transitions[activity][context]
        total_occurrences = sum(counts.values())

        probabilities = {obj: count / total_occurrences for obj, count in counts.items()}
        sorted_candidates = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)

        top_prediction, max_prob = sorted_candidates[0]
        trigger_llm = max_prob < self.threshold

        return top_prediction, max_prob, sorted_candidates, trigger_llm