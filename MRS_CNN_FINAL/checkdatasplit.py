# Import required libraries
import os
import re
import pandas as pd
from collections import defaultdict

# Define paths
recommend_music_path = 'Dataset/RecommendMusic_30'
updated_fma_small_path = 'Dataset/updated_fma_small'
metadata_path = 'Dataset/fma_metadata/tracks.csv'  # Metadata for genres

# Load metadata for genre classification
metadata = pd.read_csv(metadata_path, header=2, low_memory=False)

# Extracting the genre and track IDs from metadata
track_ids = metadata.iloc[:, 0].values
track_genres = metadata.iloc[:, 40].values
track_genre_map = {str(int(track_id)): genre for track_id, genre in zip(track_ids, track_genres)}

# Step 1: Report the number of songs in each folder
# RecommendMusic_30 folder
recommend_counts = defaultdict(int)
recommend_files = os.listdir(recommend_music_path)
for song in recommend_files:
    track_id = re.search(r'(\d+).mp3', song)
    if track_id and track_id.group(1) in track_genre_map:
        genre = track_genre_map[track_id.group(1)]
        recommend_counts[genre] += 1

# Updated fma_small folder
updated_counts = defaultdict(int)
updated_files = os.listdir(updated_fma_small_path)
for song in updated_files:
    track_id = re.search(r'(\d+).mp3', song)
    if track_id and track_id.group(1) in track_genre_map:
        genre = track_genre_map[track_id.group(1)]
        updated_counts[genre] += 1

# Print results
print("\nNumber of songs in RecommendMusic_30 folder by genre:")
if recommend_counts:
    for genre, count in recommend_counts.items():
        print(f"{genre}: {count}")
else:
    print("No songs found.")

print("\nNumber of songs in updated_fma_small folder by genre:")
if updated_counts:
    for genre, count in updated_counts.items():
        print(f"{genre}: {count}")
else:
    print("No songs found.")