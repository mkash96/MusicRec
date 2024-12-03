import torch
import os
import numpy as np
from model import CRNN
from load_data import load_dataset_pytorch
from torch.utils.data import DataLoader
from sklearn.metrics.pairwise import cosine_similarity

# Paths for saved models and features
model_path = 'Saved_Model/CRNN_Model.pth'
feature_path = 'Precomputed_Features/song_feature_averages.npy'

# Load the saved model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
n_classes = 8  # Adjust based on your dataset
img_channel, img_height, img_width = 1, 128, 128
crnn = CRNN(img_channel, img_height, img_width, n_classes)
crnn.load_state_dict(torch.load(model_path, map_location=device))
crnn.to(device)
crnn.eval()

# Check if precomputed features are already saved
if os.path.exists(feature_path):
    # Load precomputed feature averages
    song_feature_averages = np.load(feature_path, allow_pickle=True).item()
    print("Loaded precomputed feature vectors.")
else:
    # Load the test dataset and precompute feature averages
    test_dataset = load_dataset_pytorch(verbose=1, mode="Test")
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    song_feature_sums = {}
    song_slice_counts = {}
    with torch.no_grad():
        for images, label in test_loader:
            images = images.to(device)
            # Extract latent feature vectors from each slice
            features = crnn(images, return_features=True)
            features = features.cpu().numpy().flatten()

            song_name = label[0]  # Assuming label[0] is the song name

            # Accumulate the feature vectors for each song
            if song_name not in song_feature_sums:
                song_feature_sums[song_name] = features
                song_slice_counts[song_name] = 1
            else:
                song_feature_sums[song_name] += features
                song_slice_counts[song_name] += 1

    # Calculate average feature vectors per song
    song_feature_averages = {song: (feature_sum / song_slice_counts[song]) 
                             for song, feature_sum in song_feature_sums.items()}

    # Save the averaged feature vectors to disk for future use
    if not os.path.exists('Precomputed_Features'):
        os.makedirs('Precomputed_Features')

    np.save(feature_path, song_feature_averages)
    print("Precomputed feature vectors have been saved.")

# Recommendation System
print("Available songs for recommendation:")
print(list(song_feature_averages.keys()))

recommend_wrt = input("Enter the song name:\n")

if recommend_wrt not in song_feature_averages:
    print("Song not found. Please enter a valid song name.")
    exit()

# Set the anchor vector for the selected song
anchor_vector = song_feature_averages[recommend_wrt].reshape(1, -1)

# Compute cosine similarity between the anchor song and other songs
similarities = {song: cosine_similarity(anchor_vector, avg_vector.reshape(1, -1))[0][0] 
                for song, avg_vector in song_feature_averages.items() if song != recommend_wrt}

# Get top N recommendations
try:
    N = int(input("Enter the number of recommendations you'd like to see:\n"))
except ValueError:
    print("Invalid input; defaulting to 2 recommendations.")
    N = 2

top_songs = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:N]
print(f"\nTop {N} recommendations based on the song '{recommend_wrt}':")
for i, (song, similarity) in enumerate(top_songs, 1):
    print(f"{i}. Song Name: {song}, Similarity Score: {similarity:.4f}")
