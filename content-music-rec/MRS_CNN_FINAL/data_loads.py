import os
import re
import numpy as np
import cv2
from data_importer import create_spectrogram
from spectrogram_slicer import slice_spect
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

"""
Converts images and labels into training and testing matrices.
"""
def load_dataset(verbose=0, mode=None, datasetSize=1.0):
    # Check if cached data exists
    cache_dir = 'Training_Data'
    train_x_path = os.path.join(cache_dir, "train_x.npy")
    train_y_path = os.path.join(cache_dir, "train_y.npy")
    test_x_path = os.path.join(cache_dir, "test_x.npy")
    test_y_path = os.path.join(cache_dir, "test_y.npy")

    # Skip processing if cached data exists
    if mode == "Train" and all(os.path.exists(p) for p in [train_x_path, train_y_path, test_x_path, test_y_path]):
        if verbose > 0:
            print("Cached training and testing data found. Loading from cache...")
        train_x = np.load(train_x_path)
        train_y = np.load(train_y_path)
        test_x = np.load(test_x_path)
        test_y = np.load(test_y_path)

        # Genre mapping (hardcoded for consistency)
        genre = {
            "Hip-Hop": 0,
            "International": 1,
            "Electronic": 2,
            "Folk": 3,
            "Experimental": 4,
            "Rock": 5,
            "Pop": 6,
            "Instrumental": 7
        }
        return train_x, train_y, test_x, test_y, len(genre), {v: k for k, v in genre.items()}

    # Step 1: Generate spectrograms and slices if needed
    create_spectrogram(verbose, mode)
    slice_spect(verbose, mode)

    # Step 2: Validate mode and set parameters
    if mode == "Train":
        genre = {
            "Hip-Hop": 0,
            "International": 1,
            "Electronic": 2,
            "Folk": 3,
            "Experimental": 4,
            "Rock": 5,
            "Pop": 6,
            "Instrumental": 7
        }

        if verbose > 0:
            print("Compiling Training and Testing Sets ...")

        # Step 3: Load all .jpg files
        filenames = [
            os.path.join("Train_Sliced_Images", f)
            for f in os.listdir("Train_Sliced_Images")
            if f.endswith(".jpg")
        ]

        if not filenames:
            raise FileNotFoundError("No .jpg files found in Train_Sliced_Images directory.")

        images_all = [None] * len(filenames)
        labels_all = [None] * len(filenames)

        # Step 4: Process each file and extract features
        for i, f in enumerate(filenames):
            if verbose > 0 and i % 100 == 0:  # Print progress every 100 files
                print(f"Processing file {i+1}/{len(filenames)}: {f}")

            # Extract index and genre from filename
            index = int(re.search(r'Train_Sliced_Images/(.+?)_.*.jpg', f).group(1))
            genre_variable = re.search(r'Train_Sliced_Images/.*_(.+?).jpg', f).group(1)

            # Read image and convert to grayscale
            temp = cv2.imread(f, cv2.IMREAD_UNCHANGED)
            images_all[index] = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
            labels_all[index] = genre[genre_variable]

        # Step 5: Limit dataset size based on `datasetSize` parameter
        if datasetSize < 1.0:
            count_max = int(len(images_all) * datasetSize / len(genre))
            count_array = [0] * len(genre)
            images = []
            labels = []
            for i in range(len(images_all)):
                if count_array[labels_all[i]] < count_max:
                    images.append(images_all[i])
                    labels.append(labels_all[i])
                    count_array[labels_all[i]] += 1
        else:
            images = images_all
            labels = labels_all

        images = np.array(images)
        labels = np.array(labels).reshape(-1, 1)

        # Step 6: Split into training and testing sets
        train_x, test_x, train_y, test_y = train_test_split(
            images, labels, test_size=0.05, shuffle=True
        )

        # Step 7: Convert labels to one-hot vectors
        train_y = to_categorical(train_y, num_classes=len(genre))
        test_y = to_categorical(test_y, num_classes=len(genre))

        # Step 8: Cache datasets in 'Training_Data'
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        np.save(train_x_path, train_x)
        np.save(train_y_path, train_y)
        np.save(test_x_path, test_x)
        np.save(test_y_path, test_y)

        return train_x, train_y, test_x, test_y, len(genre), {v: k for k, v in genre.items()}

    if mode == "Test":
        if verbose > 0:
            print("Compiling Testing Set ...")

        # Step 3: Load all .jpg files from Test_Sliced_Images
        filenames = [
            os.path.join("Test_Sliced_Images", f)
            for f in os.listdir("Test_Sliced_Images")
            if f.endswith(".jpg")
        ]

        if not filenames:
            raise FileNotFoundError("No .jpg files found in Test_Sliced_Images directory.")

        images = []
        labels = []

        # Step 4: Process each file
        for f in filenames:
            # Extract song identifier
            song_variable = re.search(r'Test_Sliced_Images/.*_(.+?).jpg', f).group(1)
            tempImg = cv2.imread(f, cv2.IMREAD_UNCHANGED)
            images.append(cv2.cvtColor(tempImg, cv2.COLOR_BGR2GRAY))
            labels.append(song_variable)

        return np.array(images), labels
