import numpy as np
import torch
import data

def evaluate(model: torch.nn.Module, dataset: data.DMLDataset) -> float:
    """
    Returns R@1 accuracy of the given model on the given dataset.
    """
    
    all_labels = []
    all_fvecs = []
    for img, txt, label in dataset:
        all_fvecs.append(model(img).cpu().detach().numpy())
        all_labels.append(label.cpu().detach().numpy())

    all_labels = np.vstack(all_labels)
    fvecs = np.vstack(all_fvecs)
    D = np.linalg.norm(fvecs[:, np.newaxis, :] - fvecs[:, :, np.newaxis], axis=-1)
    preds = D.argmin(axis=1)
    return np.mean(preds == all_labels.ravel())
    
