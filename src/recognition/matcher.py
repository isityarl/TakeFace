import numpy as np

class Matcher:
    def __init__(self, threshold=0.75):
        self.threshold = threshold

    def match(self, known_embedding, query_embedding):
        sim = np.dot(known_embedding, query_embedding) / (
            np.linalg.norm(known_embedding) * np.linalg.norm(query_embedding)
        )
        return sim >= self.threshold, sim
