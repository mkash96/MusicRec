import os
import pandas as pd
import re
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

def create_spectrogram(verbose=0, mode=None):
    if mode == "Train":
        output_dir = 'Train_Spectogram_Images'
        if os.path.exists(output_dir):
            return
        os.makedirs(output_dir, exist_ok=True)

        filename_metadata = "Dataset/fma_metadata/tracks.csv"
        tracks = pd.read_csv(filename_metadata, header=2, low_memory=False)

        # Extract Track IDs and Genres
        tracks_id_array = tracks['track_id'].astype(str).tolist()
        tracks_genre_array = tracks.iloc[:, 40].fillna('0').astype(str).tolist()

        folder_sample = "Dataset/fma_small"
        file_names = [os.path.join(folder_sample, f) for f in os.listdir(folder_sample) if f.endswith(".mp3")]

        counter = 0
        for f in file_names:
            track_id = re.search(r'fma_small/(.+?).mp3', f).group(1).lstrip('0')
            track_index = tracks_id_array.index(track_id)
            genre = tracks_genre_array[track_index]

            if genre == '0' or not genre:
                continue

            try:
                y, sr = librosa.load(f, sr=None)
            except Exception as e:
                print(f"Failed to load file {f}: {e}")
                continue  # Skip this file if it fails to load

            melspectrogram_array = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
            mel = librosa.power_to_db(melspectrogram_array)

            # Save the spectrogram
            fig_size = plt.rcParams["figure.figsize"]
            fig_size[0] = float(mel.shape[1]) / 100
            fig_size[1] = float(mel.shape[0]) / 100
            plt.rcParams["figure.figsize"] = fig_size
            plt.axis('off')
            plt.axes([0., 0., 1., 1.0], frameon=False, xticks=[], yticks=[])
            librosa.display.specshow(mel, cmap='gray_r')

            plt.savefig(f"{output_dir}/{counter}_{genre}.jpg", bbox_inches=None, pad_inches=0)
            plt.close()
            counter += 1

        print(f"Total spectrograms generated for train set: {counter}")
        return

    elif mode == "Test":
        output_dir = 'Test_Spectogram_Images'
        os.makedirs(output_dir, exist_ok=True)

        folder_sample = "Dataset/DLMusicTest_30"
        file_names = [os.path.join(folder_sample, f) for f in os.listdir(folder_sample) if f.endswith(".mp3")]

        counter = 0
        for f in file_names:
            test_id = re.search(r'Dataset/DLMusicTest_30/(.+?).mp3', f).group(1)
            y, sr = librosa.load(f, sr=None)
            melspectrogram_array = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
            mel = librosa.power_to_db(melspectrogram_array)

            fig_size = plt.rcParams["figure.figsize"]
            fig_size[0] = float(mel.shape[1]) / 100
            fig_size[1] = float(mel.shape[0]) / 100
            plt.rcParams["figure.figsize"] = fig_size
            plt.axis('off')
            plt.axes([0., 0., 1., 1.0], frameon=False, xticks=[], yticks=[])
            librosa.display.specshow(mel, cmap='gray_r')

            plt.savefig(f"{output_dir}/{test_id}.jpg", bbox_inches=None, pad_inches=0)
            plt.close()
            counter += 1
