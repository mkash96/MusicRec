import numpy as np
from tensorflow.keras.models import Model, load_model
from data_loads import load_dataset
import os

# Load the trained model
loaded_model = load_model("Saved_Model/CNN_Model.keras")

# Modify the model to output latent features (second last layer)
matrix_size = loaded_model.layers[-2].output.shape[1]
feature_extractor = Model(inputs=loaded_model.inputs, outputs=loaded_model.layers[-2].output)

# Load and preprocess test data
images, labels = load_dataset(verbose=1, mode="Test")
images = np.expand_dims(images, axis=3)
images = images / 255.

print("Shape of input data:", images.shape)

# Initialize dictionaries to hold accumulated features and counts
feature_dict = {}
count_dict = {}

# Compute latent feature vectors
for i in range(len(labels)):
    label = labels[i]
    test_image = images[i]
    test_image = np.expand_dims(test_image, axis=0)
    feature_vector = feature_extractor.predict(test_image)[0]

    if label in feature_dict:
        feature_dict[label] += feature_vector
        count_dict[label] += 1
    else:
        feature_dict[label] = feature_vector
        count_dict[label] = 1

# Average the feature vectors for each song
for label in feature_dict:
    feature_dict[label] = feature_dict[label] / count_dict[label]

# Save the feature vectors and labels to disk
if not os.path.exists('Precomputed_Features'):
    os.makedirs('Precomputed_Features')

np.save('Precomputed_Features/feature_vectors.npy', np.array(list(feature_dict.values())))
np.save('Precomputed_Features/labels.npy', np.array(list(feature_dict.keys())))

print("Precomputed feature vectors have been saved.")
