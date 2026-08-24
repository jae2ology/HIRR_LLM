import numpy as np
from collections import defaultdict, Counter

class SequenceModel:
    """
        N-gram sequence model built from scratch

        This model estimates P(next_word | previous n-1 words)
        by counting occurrences in the training corpus.
    """
    def __init__(self, n=1, threshold=0.5):
        self.n = n
        self.threshold = threshold
        # transitions[activity][context_tuple][next_object] = count
        self.transitions = defaultdict(lambda: defaultdict(Counter))


    def train(self, dataset):
        """Train the model by counting n-grams in the dataset
            dataset format: list of tuples -> (activity, [object_1, object_2, ...])
        """
        for activity, sequence in dataset:
            for i in range(len(sequence)):
                context = tuple(sequence[i : i + 1])
                next_obj = sequence[i + 1]
                self.transitions[activity][context][next_obj] += 1

    def predict_next(self, activity, objects):
        if len(objects) < self.n:
            return None, 0.0, [], True

        context = tuple(objects[-self.n:])

        if activity not in self.transitions or context not in self.transitions[activity]:
            return None, 0.0, [], True

        counts = self.transitions[activity][context]
        total_occurrences = sum(counts.values())

        # calculate probabilites for all next objects
        probabilities = {obj: count / total_occurrences for obj, count in counts.items()}

        # rank all candidates
        sorted_candidates = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
        top_prediction, max_prob = sorted_candidates[0]

        # boolean to determine whether or not to trigger the LLM response for the HIRR model
        trigger_llm = max_prob < self.threshold

        return top_prediction, max_prob, sorted_candidates, trigger_llm