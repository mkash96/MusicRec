import numpy as np
import keras
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras import optimizers
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, Callback
from data_loads import load_dataset
import pandas as pd
import tensorflow as tf
import librosa.display
import random
import traceback
import sys

# Step 1: GPU Configuration
print("Step 1: Checking GPU availability...")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU is available and memory growth is enabled.")
    except RuntimeError as e:
        print("Error while enabling GPU memory growth:", e)
else:
    print("No GPU detected, running on CPU.")

# Step 2: Set random seed
print("Step 2: Setting random seed...")
tf.random.set_seed(1)

# Step 3: Load dataset
print("Step 3: Loading dataset...")
train_x, train_y, test_x, test_y, n_classes, genre = load_dataset(verbose=1, mode="Train", datasetSize=0.75)
print(f"Dataset loaded. Training samples: {train_x.shape[0]}, Test samples: {test_x.shape[0]}")

# Step 4: Expand dimensions
print("Step 4: Expanding dimensions...")
train_x = train_x.reshape(train_x.shape[0], train_x.shape[1], train_x.shape[2], 1)
test_x = test_x.reshape(test_x.shape[0], test_x.shape[1], test_x.shape[2], 1)
print(f"Training data shape: {train_x.shape}, Test data shape: {test_x.shape}")

# Step 5: Splitting validation data
print("Step 5: Splitting validation data...")
val_split = 0.1
split_idx = int(len(train_x) * (1 - val_split))
val_x, val_y = train_x[split_idx:], train_y[split_idx:]
train_x, train_y = train_x[:split_idx], train_y[:split_idx]
print(f"Validation samples: {val_x.shape[0]}, Training samples: {train_x.shape[0]}")

# Step 6: Creating TensorFlow datasets with normalization
print("Step 6: Creating TensorFlow datasets with normalization...")
def preprocess(x, y):
    x = tf.cast(x, tf.float32) / 255.0  # Normalize data
    return x, y

# Create TensorFlow datasets with normalization
train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y)).map(preprocess).cache()
val_dataset = tf.data.Dataset.from_tensor_slices((val_x, val_y)).map(preprocess).cache()

# Shuffle, batch, and prefetch datasets
train_dataset = train_dataset.shuffle(buffer_size=1000).batch(32).prefetch(tf.data.AUTOTUNE)
val_dataset = val_dataset.batch(32).prefetch(tf.data.AUTOTUNE)

print("TensorFlow datasets created with normalization.")

# Step 7: Define CNN model
print("Step 7: Defining the CNN model...")
model = Sequential([
    Conv2D(filters=64, kernel_size=(3, 3), activation="relu", input_shape=(128, 128, 1), kernel_initializer='he_normal'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2), strides=2),
    Conv2D(filters=128, kernel_size=(3, 3), activation="relu", kernel_initializer='he_normal'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2), strides=2),
    Conv2D(filters=256, kernel_size=(3, 3), activation="relu", kernel_initializer='he_normal'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2), strides=2),
    Conv2D(filters=512, kernel_size=(3, 3), activation="relu", kernel_initializer='he_normal'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2), strides=2),
    Flatten(),
    BatchNormalization(),
    Dropout(0.5),
    Dense(512, activation="relu", kernel_initializer='he_normal'),
    Dropout(0.5),
    Dense(256, activation="relu", kernel_initializer='he_normal'),
    Dropout(0.3),
    Dense(128, activation="relu", kernel_initializer='he_normal'),
    Dense(n_classes, activation="softmax")
])
print(model.summary())

# Step 8: Compile the model
print("Step 8: Compiling the model...")
initial_learning_rate = 0.0001
model.compile(
    loss="categorical_crossentropy",
    optimizer=optimizers.Adam(learning_rate=initial_learning_rate),
    metrics=['accuracy']
)

# Step 9: Define callbacks
print("Step 9: Setting up callbacks...")
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-6, verbose=1)
checkpoint = ModelCheckpoint('Saved_Model/best_model.keras', monitor='val_loss', save_best_only=True, mode='min', verbose=1)

# Custom callback to calculate and store F1 score after each epoch
class F1ScoreCallback(Callback):
    def __init__(self, validation_data):
        self.validation_data = validation_data
        self.f1_scores = []

    def on_epoch_end(self, epoch, logs=None):
        val_x, val_y = self.validation_data
        y_pred = self.model.predict(val_x)
        y_pred_classes = tf.argmax(y_pred, axis=1)
        y_true = tf.argmax(val_y, axis=1)
        
        # Calculate weighted F1 score
        f1 = f1_score(y_true, y_pred_classes, average='weighted')
        self.f1_scores.append(f1)

f1_callback = F1ScoreCallback(validation_data=(val_x, val_y))

# Step 10: Train the model
print("Step 10: Starting training...")
history = model.fit(
    train_dataset,
    epochs=100,
    validation_data=val_dataset,
    callbacks=[early_stopping, reduce_lr, checkpoint, f1_callback]
)

# Step 11: Save and evaluate
print("Step 11: Saving and evaluating...")
try:
    model.save("Saved_Model/CNN_Model.keras")
    # Evaluate using batches to reduce memory usage
    print("Evaluating the model on test data...")
    test_dataset = tf.data.Dataset.from_tensor_slices((test_x, test_y)).map(preprocess).batch(32).prefetch(tf.data.AUTOTUNE)
    score = model.evaluate(test_dataset, verbose=1)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])
except Exception as e:
    print("An error occurred during saving and evaluation:")
    traceback.print_exc()
    sys.exit(1)

# Continue with the rest of the code inside a try-except block
try:
    # Save training history
    pd.DataFrame(history.history).to_csv("Saved_Model/CNN_training_history.csv")

    # Plot Accuracy and Loss Graphs
    plt.figure(figsize=(12, 5))

    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(history.epoch, history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.epoch, history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.grid(True)

    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(history.epoch, history.history['loss'], label='Training Loss')
    plt.plot(history.epoch, history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.grid(True)

    # Save the figure
    plt.tight_layout()
    plt.savefig('accuracy_loss_graphs.png', dpi=300)

    # Show the figure
    plt.show()

    # Plot F1 Score Over Epochs
    print("Plotting F1 Score Over Epochs...")
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(f1_callback.f1_scores) + 1), f1_callback.f1_scores, label='Validation F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title('Validation F1 Score Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig("Saved_Model/Validation_F1_Score.png", bbox_inches='tight')
    plt.show()

    # Predictions on the test set
    print("Generating predictions on the test set...")
    # Use test_dataset to reduce memory usage
    y_pred = model.predict(test_dataset)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(test_y, axis=1)

    # Compute F1 Score
    f1 = f1_score(y_true, y_pred_classes, average='weighted')
    print(f'Weighted F1 Score: {f1:.4f}')

    # Confusion Matrix
    print("Creating confusion matrix...")
    conf_matrix = confusion_matrix(y_true, y_pred_classes)

    target_names = [genre[key] for key in sorted(genre.keys())]  # Extract genre names in sorted order

    # Plot Confusion Matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig("Saved_Model/CNN_Confusion_Matrix_With_Labels.png")
    plt.show()

    # Classification Report
    print("\nClassification Report:\n", classification_report(y_true, y_pred_classes, target_names=target_names))

    # Plot spectrograms with predicted and actual labels
    print("Plotting spectrogram predictions...")
    plt.figure(figsize=(20, 10))
    indices = list(range(len(test_x)))
    random.shuffle(indices)
    num_samples = 10  # Adjust the number of spectrograms to display

    for i, idx in enumerate(indices[:num_samples]):
        plt.subplot(2, num_samples // 2, i + 1)
        img = test_x[idx].squeeze()
        img = img * 255.0
        img_db = librosa.amplitude_to_db(img, ref=np.max)
        librosa.display.specshow(img_db, sr=22050, cmap='viridis')
        true_label = genre[y_true[idx]]  
        pred_label = genre[y_pred_classes[idx]]  
        plt.title(f"Predicted: {pred_label}\nActual: {true_label}", fontsize=10)
        plt.axis('off')

    plt.tight_layout()
    plt.savefig("Saved_Model/spectrogram_predictions.png", bbox_inches='tight')  
    plt.show()

except Exception as e:
    print("An error occurred during post-processing:")
    traceback.print_exc()
    sys.exit(1)

print("Script execution completed.")