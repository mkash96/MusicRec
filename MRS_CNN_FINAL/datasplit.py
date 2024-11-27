# Import required libraries
import os
import shutil
import random
import json
import re
from collections import defaultdict

# Define paths
fma_small_path = 'Dataset/fma_small'
recommend_music_path = 'Dataset/RecommendMusic_30'
updated_fma_small_path = 'Dataset/updated_fma_small'
metadata_path = 'Dataset/fma_metadata/tracks.csv'  # Metadata for genres

# Load metadata for genre classification
import pandas as pd
metadata = pd.read_csv(metadata_path, header=2, low_memory=False)

# Extracting the genre and track IDs from metadata
track_ids = metadata.iloc[:, 0].values
track_genres = metadata.iloc[:, 40].values
track_ids = [str(int(x)) for x in track_ids]

# Step 1: Separate 20% of the dataset, ensuring equal distribution across genres

# Grouping songs by genre
songs_by_genre = defaultdict(list)

# Iterate over fma_small directory to find songs and match with track IDs
subdirectories = [d for d in os.listdir(fma_small_path) if os.path.isdir(os.path.join(fma_small_path, d))]
for d in subdirectories:
    genre_directory = os.path.join(fma_small_path, d)
    audio_files = [os.path.join(genre_directory, f) for f in os.listdir(genre_directory) if f.endswith(".mp3")]

    for f in audio_files:
        track_id_match = re.search(r'fma_small/.*/(.+?).mp3', f)
        if track_id_match:
            track_id = int(track_id_match.group(1))
        else:
            continue

        try:
            track_index = track_ids.index(str(track_id))
        except ValueError:
            print(f"Track ID {track_id} not found in metadata, skipping...")
            continue

        genre = track_genres[track_index]
        if str(genre) != '0':
            songs_by_genre[genre].append(f)

# Debugging: Print number of songs found for each genre
for genre, songs in songs_by_genre.items():
    print(f"Genre: {genre}, Number of songs found: {len(songs)}")

# Check if songs are available in each genre
if not any(songs_by_genre.values()):
    print("No songs found in the dataset. Please check the dataset paths and metadata.")
else:
    # Calculate 20% of songs per genre
    songs_to_recommend = []
    songs_moved_to_recommend = defaultdict(int)
    for genre, songs in songs_by_genre.items():
        if len(songs) > 0:
            num_recommend_songs = max(1, len(songs) // 5)
            random.shuffle(songs)
            songs_to_recommend.extend(songs[:num_recommend_songs])
            songs_moved_to_recommend[genre] = num_recommend_songs
            songs_by_genre[genre] = songs[num_recommend_songs:]

    # Debugging: Print number of songs selected for recommendation
    print(f"Total number of songs selected for recommendation: {len(songs_to_recommend)}")

    # Step 2: Move 20% of songs to RecommendMusic_30 and remove from fma_small
    os.makedirs(recommend_music_path, exist_ok=True)
    os.makedirs(updated_fma_small_path, exist_ok=True)

    if not songs_to_recommend:
        print("No songs selected for recommendation. Please check the dataset.")
    else:
        for song in songs_to_recommend:
            song_name = os.path.basename(song)
            dest_path = os.path.join(recommend_music_path, song_name)
            if not os.path.exists(dest_path):
                shutil.move(song, dest_path)
            else:
                print(f"Song {song_name} already exists in RecommendMusic_30. Skipping...")

    # Step 3: Move the remaining songs to the updated fma_small folder
    songs_moved_to_updated = defaultdict(int)
    for genre, songs in songs_by_genre.items():
        for song in songs:
            song_name = os.path.basename(song)
            dest_path = os.path.join(updated_fma_small_path, song_name)
            if not os.path.exists(dest_path):
                shutil.move(song, dest_path)
                songs_moved_to_updated[genre] += 1
            else:
                print(f"Song {song_name} already exists in updated_fma_small. Skipping...")

    # Step 4: Report the number of songs in each folder by genre
    print("\nNumber of songs in RecommendMusic_30 folder by genre:")
    for genre, count in songs_moved_to_recommend.items():
        print(f"{genre}: {count}")

    print("\nNumber of songs in updated_fma_small folder by genre:")
    for genre, count in songs_moved_to_updated.items():
        print(f"{genre}: {count}")

    print("Data separation complete. 20% of the songs have been moved to RecommendMusic_30.")
