# evaluation.py

import torch
from sklearn.metrics import f1_score

def predict_top_k(model, dataloader, device, k=2):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.topk(outputs, k, dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(targets.cpu().tolist())
    return all_preds, all_targets

def compute_top_k_f1_score(model, dataloader, device, k=2):
    all_preds_k, all_targets = predict_top_k(model, dataloader, device, k)
    correct_preds = []
    for preds, target in zip(all_preds_k, all_targets):
        if target in preds:
            correct_preds.append(target)
        else:
            correct_preds.append(preds[0])  # Default to top prediction
    score = f1_score(all_targets, correct_preds, average='weighted')
    return score
