# load_data.py

import os
import re
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
from import_data import create_spectrogram
from slice_spectrogram import slice_spect

class MusicDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels  # Keep labels as-is (strings for Test, integers for Train)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        # Convert image to torch tensor and add channel dimension
        image = torch.from_numpy(image).unsqueeze(0).float()  # shape: (1, H, W)
        
        # Convert label to tensor if it's an integer, otherwise return as is (string for song names in test set)
        if isinstance(label, (int, np.integer)):
            label = torch.tensor(label).long()
        
        return image, label

def load_dataset_pytorch(verbose=0, mode=None, datasetSize=1.0):
    create_spectrogram(verbose, mode)
    slice_spect(verbose, mode)
    
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
            
        filenames = [os.path.join("Train_Sliced_Images", f) for f in os.listdir("Train_Sliced_Images") if f.endswith(".jpg")]
        images_all = []
        labels_all = []
        
        for f in filenames:
            genre_variable = re.search(r'Train_Sliced_Images/.*_(.+?).jpg', f).group(1)
            temp = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
            images_all.append(temp)
            labels_all.append(genre[genre_variable])

        images_all = np.array(images_all)
        labels_all = np.array(labels_all)

        # If datasetSize is less than 1.0, sample a subset
        if datasetSize < 1.0:
            from sklearn.model_selection import train_test_split
            images_all, _, labels_all, _ = train_test_split(
                images_all, labels_all, train_size=datasetSize, stratify=labels_all, random_state=42)
        
        # Split into training and validation sets
        from sklearn.model_selection import train_test_split
        train_images, val_images, train_labels, val_labels = train_test_split(
            images_all, labels_all, test_size=0.1, stratify=labels_all, random_state=42)

        # Create PyTorch datasets
        train_dataset = MusicDataset(train_images, train_labels)
        val_dataset = MusicDataset(val_images, val_labels)

        n_classes = len(genre)
        genre_new = {value: key for key, value in genre.items()}

        return train_dataset, val_dataset, n_classes, genre_new

    elif mode == "Test":
        if verbose > 0:
            print("Compiling Test Set ...")
            
        filenames = [os.path.join("Test_Sliced_Images", f) for f in os.listdir("Test_Sliced_Images") if f.endswith(".jpg")]
        images = []
        labels = []
        
        for f in filenames:
            song_variable = re.search(r'Test_Sliced_Images/.*_(.+?).jpg', f).group(1)
            tempImg = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
            images.append(tempImg)
            labels.append(song_variable)  # Keep song names as labels for recommendation

        images = np.array(images)
        labels = np.array(labels, dtype=object)  # Explicitly ensure labels are strings

        test_dataset = MusicDataset(images, labels)
        return test_dataset
