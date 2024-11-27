import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from load_data import load_dataset_pytorch
from model import CRNN
from utils import EarlyStopping, train_batch, evaluate
from evaluation import compute_top_k_f1_score
from visualization import plot_confusion_matrix, create_spectrogram_plots
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

def get_predictions(model, loader, device):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    return np.array(all_preds), np.array(all_targets)

def train_model(epochs=100, initial_epoch=0):
    # Ensure necessary directories exist
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('Saved_Model'):
        os.makedirs('Saved_Model')

    # Load dataset
    train_dataset, val_dataset, n_classes, genre_new = load_dataset_pytorch(verbose=1, mode="Train", datasetSize=0.75)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # Initialize model
    # device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    device = torch.device('cpu')
    img_channel, img_height, img_width = 1, 128, 128  # Adjust based on your images
    crnn = CRNN(img_channel, img_height, img_width, n_classes)
    crnn.to(device)
    
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(crnn.parameters(), lr=1e-4)  # Starting with a standard learning rate
    
    # Initialize the learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, min_lr=1e-6)
    
    # Initialize EarlyStopping
    early_stopping = EarlyStopping(patience=10, verbose=True, path='checkpoints/last.pt')
    
    # Training loop
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    for epoch in range(initial_epoch, initial_epoch + epochs):
        print(f'Epoch {epoch+1}/{initial_epoch + epochs}')
        total_loss = 0
        crnn.train()
        for data in train_loader:
            loss = train_batch(crnn, data, optimizer, criterion, device)
            total_loss += loss
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        evaluation = evaluate(crnn, val_loader, criterion, device)
        val_loss = evaluation['loss']
        val_acc = evaluation['acc']
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        print(f'Training Loss: {avg_train_loss:.4f}, Validation Loss: {val_loss:.4f}, Validation F1 Score: {val_acc:.4f}')
        
        # Step the scheduler
        scheduler.step(val_loss)
        
        # Print the current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Current Learning Rate: {current_lr:.8f}')

        # Early stopping check
        early_stopping(val_loss, crnn)
        if early_stopping.early_stop:
            print("Early stopping")
            break

        # Save the model checkpoint
        torch.save(crnn.state_dict(), 'checkpoints/last.pt')
        # Save optimizer state for future resuming
        torch.save(optimizer.state_dict(), 'checkpoints/optimizer.pt')
        # Save scheduler state
        torch.save(scheduler.state_dict(), 'checkpoints/scheduler.pt')
        # Save early stopping state
        torch.save(early_stopping.state_dict(), 'checkpoints/early_stopping.pt')

    # Load the best model
    crnn.load_state_dict(torch.load('checkpoints/last.pt'))
    crnn.eval()
    
    # Save the final model
    torch.save(crnn.state_dict(), 'Saved_Model/CRNN_Model.pth')

    # Plot Losses
    plt.figure(figsize=(10, 5))
    plt.plot(range(initial_epoch + 1, initial_epoch + len(train_losses) + 1), train_losses, label='Training Loss')
    plt.plot(range(initial_epoch + 1, initial_epoch + len(val_losses) + 1), val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Plot Validation F1 Score
    plt.figure(figsize=(10, 5))
    plt.plot(range(initial_epoch + 1, initial_epoch + len(val_accuracies) + 1), val_accuracies, label='Validation F1 Score')
    plt.title('Validation F1 Score Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Evaluate the model
    val_preds, val_targets = get_predictions(crnn, val_loader, device)
    
    # Compute the confusion matrix
    cm = confusion_matrix(val_targets, val_preds)
    
    # Get class names
    class_names = [genre_new[i] for i in range(n_classes)]
    
    # Plot confusion matrix
    plot_confusion_matrix(cm, target_names=class_names, title='Confusion Matrix', normalize=False)
    
    # Classification report
    print("\nClassification Report:\n")
    print(classification_report(val_targets, val_preds, target_names=class_names))
    
    # Compute Top 2 and Top 3 F1 Scores
    top2_f1 = compute_top_k_f1_score(crnn, val_loader, device, k=2)
    print(f'Top 2 F1 Score: {top2_f1:.4f}')
    
    top3_f1 = compute_top_k_f1_score(crnn, val_loader, device, k=3)
    print(f'Top 3 F1 Score: {top3_f1:.4f}')
    
    # Plot sample spectrograms
    data_iter = iter(train_loader)
    images, labels = next(data_iter)
    create_spectrogram_plots(images.cpu(), labels.cpu(), class_names, num_classes=n_classes)

if __name__ == '__main__':
    train_model(epochs=100)  # Set the number of epochs you want to train for
