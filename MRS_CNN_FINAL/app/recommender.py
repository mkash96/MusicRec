import numpy as np

class Recommender:
    def __init__(self, feature_vectors_path, labels_path):
        self.feature_vectors = np.load(feature_vectors_path, allow_pickle=True)
        self.labels = np.load(labels_path, allow_pickle=True).astype(str)

    def get_song_list(self):
        return self.labels

    def recommend_songs(self, seed_song, top_n=2, exclude_songs=None):
        if exclude_songs is None:
            exclude_songs = []

        if seed_song not in self.labels:
            raise ValueError(f"Song '{seed_song}' not found in the dataset.")

        seed_index = np.where(self.labels == seed_song)[0][0]
        seed_vector = self.feature_vectors[seed_index]

        similarities = []
        for i, song_vector in enumerate(self.feature_vectors):
            song_label = self.labels[i]
            if song_label != seed_song and song_label not in exclude_songs:
                similarity = np.dot(seed_vector, song_vector) / (
                    np.linalg.norm(seed_vector) * np.linalg.norm(song_vector)
                )
                similarities.append((song_label, float(similarity)))

        # Sort by similarity score
        similarities = sorted(similarities, key=lambda x: x[1], reverse=True)

        return similarities if top_n is None else similarities[:top_n]