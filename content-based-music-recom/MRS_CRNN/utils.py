# utils.py

import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score

def train_batch(crnn, data, optimizer, criterion, device):
    crnn.train()
    images, targets = [d.to(device) for d in data]
    optimizer.zero_grad()
    outputs = crnn(images)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
    return loss.item()

def evaluate(crnn, dataloader, criterion, device):
    crnn.eval()
    total_loss = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data in dataloader:
            images, targets = [d.to(device) for d in data]
            outputs = crnn(images)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    acc = f1_score(all_targets, all_preds, average='weighted')
    return {'loss': avg_loss, 'acc': acc}

class EarlyStopping:
    def __init__(self, patience=10, verbose=True, delta=0, path='checkpoints/last.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

    def state_dict(self):
        return {
            'counter': self.counter,
            'best_score': self.best_score,
            'early_stop': self.early_stop
        }

    def load_state_dict(self, state_dict):
        self.counter = state_dict['counter']
        self.best_score = state_dict['best_score']
        self.early_stop = state_dict['early_stop']