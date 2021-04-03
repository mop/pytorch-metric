import numpy as np
import torch
import data
from scipy.spatial.distance import cdist

def evaluate(model: torch.nn.Module, dataset: data.DMLDataset) -> float:
    """
    Returns R@1 accuracy of the given model on the given dataset.
    """
    
    model_is_training = model.training
    model.eval()

    all_labels = []
    all_fvecs = []
    for batch in dataset:
        img, txt, label = batch['image'], batch['text'], batch['label']
        all_fvecs.append(model(img.cuda()).cpu().detach().numpy())
        all_labels.append(label.cpu().detach().numpy())

    all_labels = np.concatenate(all_labels)
    fvecs = np.vstack(all_fvecs)
    fvecs = fvecs / np.maximum(1e-5, np.linalg.norm(fvecs, axis=-1, keepdims=True))

    D = cdist(fvecs, fvecs)
    D += np.eye(D.shape[0]) * 1000
    preds = all_labels[D.argmin(axis=1)]
    print(D.shape, fvecs.shape, all_labels.shape)
    print(preds.shape)

    model.train()
    model.train(model_is_training)

    return np.mean(preds == all_labels.ravel())
    
