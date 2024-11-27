# visualization.py

import matplotlib.pyplot as plt
import numpy as np
import itertools
import librosa.display
from sklearn.utils import shuffle

def plot_confusion_matrix(cm, target_names, title='Confusion Matrix', cmap=None, normalize=False):
    """
    Plots a confusion matrix using matplotlib.

    Args:
        cm (array, shape = [n, n]): Confusion matrix
        target_names (list): Class names
        title (str): Title of the plot
        cmap: Color map
        normalize (bool): Normalize the confusion matrix

    Returns:
        None
    """
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Wistia')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=45, fontsize=8)
    plt.yticks(tick_marks, target_names, fontsize=8)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    threshold = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            value = "{:0.2f}".format(cm[i, j])
        else:
            value = "{:,}".format(cm[i, j])
        plt.text(j, i, value,
                 horizontalalignment="center",
                 color="white" if cm[i, j] > threshold else "black",
                 fontsize=8)

    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label\nAccuracy={:0.4f}; Misclassification Rate={:0.4f}'.format(accuracy, misclass))
    plt.show()

def create_spectrogram_plots(X, Y, class_labels, num_classes=8):
    """
    Plots sample spectrograms for each class.

    Args:
        X (tensor): Images tensor
        Y (tensor): Labels tensor
        class_labels (list): List of class names
        num_classes (int): Number of classes

    Returns:
        None
    """
    X = X.numpy()
    Y = Y.numpy()
    X, Y = shuffle(X, Y)
    fig, axes = plt.subplots(2, 4, figsize=(15, 7))
    axes = axes.flatten()
    class_counts = {i: 0 for i in range(num_classes)}
    for img, label in zip(X, Y):
        if class_counts[label] < 1:
            idx = sum(class_counts.values()) - 1
            img_db = librosa.amplitude_to_db(img.squeeze(), ref=np.max)
            librosa.display.specshow(img_db, sr=22050, ax=axes[idx], cmap='viridis')
            axes[idx].set_title(class_labels[label])
            axes[idx].axis('off')
            class_counts[label] += 1
        if all(count >= 1 for count in class_counts.values()):
            break
    plt.tight_layout()
    plt.show()
