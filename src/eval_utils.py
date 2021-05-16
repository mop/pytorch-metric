import numpy as np
import torch
import data
import torch.nn.functional as F
from scipy.spatial.distance import cdist


def extract_features(model: torch.nn.Module,
                     dataset: data.DMLDataset,
                     use_cuda: bool = False) -> (np.ndarray, np.ndarray):
    """
    Extracts all feature vectors from the dataset with the model.

    :model: Is a torch embedding ensemble model
    :dataset: stores the dataset
    :returns: Returns a tuple of (feature_vectors, labels)
    """
    model_is_training = model.training
    model.eval()

    all_labels = []
    all_fvecs = []
    if not use_cuda:
        for batch in dataset:
            img, txt, label = batch['image'], batch['text'], batch['label']
            fvecs = [f.cpu().detach().numpy() for f in model(img.cuda())]
            tmp = []
            weights = [3, 1, 2]
            weights = [1,1,1]
            for w, f in zip(weights, fvecs):
                f = (w * f) / np.maximum(1e-5, np.linalg.norm(f, axis=-1, keepdims=True))
                tmp.append(f)
            #__import__('pdb').set_trace()
            fvecs = np.concatenate(tmp, axis=1)
            all_fvecs.append(fvecs)
            if len(label.size()) > 1:
                label = torch.argmax(label, dim=1)
            all_labels.append(label.cpu().detach().numpy())

        all_labels = np.concatenate(all_labels)
        fvecs = np.vstack(all_fvecs)
        fvecs = fvecs / np.maximum(1e-5, np.linalg.norm(fvecs, axis=-1, keepdims=True))
    else:
        for batch in dataset:
            img, txt, label = batch['image'], batch['text'], batch['label']
            fvecs = model(img.cuda())
            tmp = []
            weights = [3, 1, 2]
            weights = [2,0,1]
            for w, f in zip(weights, fvecs):
                f = w * F.normalize(f, p=2, dim=1, eps=1e-5)
                tmp.append(f)
            fvecs = torch.cat(tmp, axis=1)
            all_fvecs.append(fvecs.cpu().detach().numpy())
            if len(label.size()) > 1:
                label = torch.argmax(label, dim=1)
            all_labels.append(label.cpu().detach().numpy())

        all_labels = np.concatenate(all_labels)
        fvecs = np.vstack(all_fvecs)
        fvecs = fvecs / np.maximum(1e-5, np.linalg.norm(fvecs, axis=-1, keepdims=True))

    model.train()
    model.train(model_is_training)

    return fvecs, all_labels


def evaluate(model: torch.nn.Module, dataset: data.DMLDataset, use_cuda: bool = False) -> float:
    """
    Returns R@1 accuracy of the given model on the given dataset.
    """
    
    fvecs, all_labels = extract_features(model, dataset, use_cuda=use_cuda)

    if not use_cuda:
        D = cdist(fvecs, fvecs)
    else:
        torch_fvecs = torch.from_numpy(fvecs).cuda()
        Ds = []
        for i in range(len(fvecs)):
            G_chunk = torch.mm(torch_fvecs[i, :].view(1, -1), torch.t(torch_fvecs))
            G_chunk = torch.sqrt(torch.relu(2 - 2 * G_chunk)).cpu().detach().numpy()
            Ds.append(G_chunk)
        Ds = np.vstack(Ds)
        D = Ds

    D += np.eye(D.shape[0]) * 1000
    preds = all_labels[D.argmin(axis=1)]
    print(D.shape, fvecs.shape, all_labels.shape)
    print(preds.shape)

    return np.mean(preds == all_labels.ravel())


def f1_evaluate(model: torch.nn.Module,
                dataset: data.DMLDataset,
                use_cuda: bool=False,
                threshold: float = None) -> (float, float):
    """
    Performs the Kaggle evaluation on the validation set (by computing the harmonic mean). 

    :model: The DML model
    :dataset: validation dataset
    :threshold: The similarity threshold (by which we determine if something is a pair)
                If threshold is None, we try to find the optimal one by linearly trying out
                10 thresholds from 0 to the max distance
    :returns: A tuple of (threshold, f1_score)
    """
    fvecs, all_labels = extract_features(model, dataset, use_cuda=use_cuda)

    #D = cdist(fvecs, fvecs)
    if not use_cuda:
        D = cdist(fvecs, fvecs)
    else:
        torch_fvecs = torch.from_numpy(fvecs).cuda()
        Ds = []
        for i in range(len(fvecs)):
            G_chunk = torch.mm(torch_fvecs[i, :].view(1, -1), torch.t(torch_fvecs))
            G_chunk = torch.clip(torch.sqrt(torch.relu(2.0 - 2.0 * G_chunk)), 0, np.sqrt(2)).cpu().detach().numpy()
            Ds.append(G_chunk)
        Ds = np.vstack(Ds)
        D = Ds

    #D += np.eye(D.shape[0]) * 1000

    # reset model to training
    print(D.max(), np.sqrt(2))
    thresholds = [threshold] if threshold is not None else np.linspace(0.5, np.sqrt(2), 10)
    label_counts = np.bincount(all_labels)

    threshold_scores = []
    for t in thresholds:
        preds = D <= t
        f1_scores = []
        for i, pred in enumerate(preds):
            my_lbl = all_labels[i]
            other_lbl = all_labels[pred]
            num_tp = (my_lbl == other_lbl).sum()
            num_labels = label_counts[my_lbl]

            precision = num_tp / np.maximum(1, float(len(other_lbl)))
            recall = num_tp / float(num_labels)

            f1_score = 2 * precision * recall / np.maximum(1e-5, (precision + recall))
            f1_scores.append(f1_score)

        threshold_scores.append(np.mean(f1_scores))
    best_ix = np.argmax(threshold_scores)
    print(thresholds, threshold_scores)
    return thresholds[best_ix], threshold_scores[best_ix]
