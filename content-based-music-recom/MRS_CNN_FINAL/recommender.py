# recommendation.py

import numpy as np
from tensorflow.keras.models import Model, load_model
import os

# Load the precomputed feature vectors and labels
feature_vectors = np.load('Precomputed_Features/feature_vectors.npy', allow_pickle=True)
labels = np.load('Precomputed_Features/labels.npy', allow_pickle=True)

# Ensure labels are in string format
labels = labels.astype(str)

# Display list of available test songs
print("Available songs for recommendation:")
print(labels)

# Enter a song name which will be an seed song
recommend_wrt = input("Enter Song name:\n")

# Check if the entered song exists
if recommend_wrt not in labels:
    print("Song not found. Please enter a valid song name.")
    exit()

# Retrieve the seed song's feature vector
seed_index = np.where(labels == recommend_wrt)[0][0]
prediction_seed = feature_vectors[seed_index]

# Ensure prediction_seed is a 1D array
prediction_seed = prediction_seed.flatten()

# Compute cosine similarity with other songs
distance_array = []
for i in range(len(labels)):
    if labels[i] != recommend_wrt:
        # Ensure feature_vectors[i] is a 1D array
        song_vector = feature_vectors[i].flatten()

        # Cosine Similarity
        numerator = np.dot(prediction_seed, song_vector)
        denominator = np.linalg.norm(prediction_seed) * np.linalg.norm(song_vector)
        similarity = numerator / denominator

        # Ensure similarity is a scalar float
        similarity = float(similarity)

        distance_array.append((similarity, labels[i]))


# Sort songs based on similarity
distance_array.sort(reverse=True, key=lambda x: x[0])

# Number of recommendations
num_recommendations = 2  # You can adjust this number

print("Recommendations based on the song '{}':".format(recommend_wrt))
for i in range(num_recommendations):
    print("Song Name: {} with similarity score: {:.4f}".format(distance_array[i][1], distance_array[i][0]))
