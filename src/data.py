import torch
import csv
import torch.utils.data as data
import pandas as pd
import numpy as np
import os
from PIL import Image
import torchvision.transforms as transforms
import glob

def onehot(label: np.ndarray, n_classes: int) -> torch.Tensor:
    """
    Returns a one-hot encoding of the labels
    """
    return torch.zeros(label.size(0), n_classes).scatter_(
                    1, label.view(-1, 1), 1)


def mixup(data: torch.Tensor, targets: torch.Tensor, alpha: float, n_classes: int) -> (np.ndarray, np.ndarray):
    indices = torch.randperm(data.size(0))
    data2 = data[indices]
    targets2 = targets[indices]

    targets = onehot(targets, n_classes)
    targets2 = onehot(targets2, n_classes)

    lam = torch.FloatTensor([np.random.beta(alpha, alpha)])
    data = data * lam + data2 * (1 - lam)
    targets = targets * lam + targets2 * (1 - lam)

    return data, targets


class DMLDataset(data.Dataset):
    """
    This class is responsible for loading the data for deep metric learning.
    """
    def __init__(self,
                 csv_path: str,
                 image_size: int = 224,
                 mixup_alpha: float = 0.0,
                 is_training: bool = True,
                 is_testing: bool = False,
                 subset_labels=None):
        """
        Initializes the DML dataset.

        Arguments:
            csv_path: Path to the training csv file.
            image_size: size of the images (after pre-processing)
            mixup_alpha: mixup fraction
            is_training: True if it is training (and augmentation should be performed)
            is_testing: True if it is testing (images will be loaded from the test folder)
            subset_labels: Subset of labels to load for training (if None, all are loaded)
        """
        self.csv_path = csv_path
        self.image_size = image_size
        self.mixup_alpha = mixup_alpha
        self.is_training = is_training
        self.is_testing = is_testing
        self.subset_labels = subset_labels

        self.df = pd.read_csv(csv_path)

        if not self.is_testing:
            labels = self.df['label_group'].unique()
        else:
            labels = []
        if self.subset_labels is not None:
            self.subset_labels = set(self.subset_labels)
            self.num_classes = len(self.subset_labels)
        else:
            self.num_classes = len(labels)

        self.files = []
        self.labels = []
        self.texts = []
        self.posting_ids = []
        if not self.is_testing:
            i = 0
            for lbl in labels:
                if self.subset_labels is not None and lbl not in self.subset_labels:
                    continue
                relevant = self.df[self.df['label_group'] == lbl]
                self.labels += [i] * len(relevant)
                self.files += list(relevant['image'])
                self.texts += list(relevant['title'])
                self.posting_ids += list(relevant['posting_id'])
                i += 1
            self.labels = np.asarray(self.labels)
        else:
            for i in range(len(self.df)):
                self.posting_ids.append(self.df['posting_id'].iloc[i])
                self.files.append(self.df['image'].iloc[i])
                self.texts.append(self.df['title'].iloc[i])
                self.labels.append(-1)
            self.labels = np.asarray(self.labels)

        self.root = os.path.dirname(self.csv_path) + '/train_images/'
        if self.is_testing:
            self.root = os.path.dirname(self.csv_path) + '/test_images/'

        normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        if self.is_training:
            self.tform = transforms.Compose([
                transforms.RandomResizedCrop(self.image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ])
        else:
            self.tform = transforms.Compose([
                transforms.CenterCrop(self.image_size),
                transforms.ToTensor(),
                normalize
            ])

    def collate_fn(self, elements: list) -> dict:
        """
        Collates (pre-processes) a list of elements to a batch.

        Arguments:
            :elements: is a list of dicts which keys image, text and label.
        Returns:
            A dict with keys image text and label containing the batched/preprocessed
            versions of the text, label and image.
        """
        images = torch.zeros((len(elements), 3, self.image_size, self.image_size), dtype=torch.float32)
        labels = torch.zeros((len(elements),), dtype=torch.int64)
        text = []
        posting_ids = []

        for i in range(len(elements)):
            if not self.is_training:
                images[i, ...] = self.tform(elements[i]['image'])
            else:
                images[i, ...] = elements[i]['image']
            text.append(elements[i]['text'])
            posting_ids.append(elements[i]['posting_id'])
            if not self.is_testing:
                labels[i] = int(elements[i]['label'])
        if not self.is_testing:
            if self.mixup_alpha > 0 and self.is_training:
                images, labels = mixup(images, labels, self.mixup_alpha, self.num_classes)
            else:
                labels = onehot(labels, self.num_classes)

        return { 'image': images, 'text': text, 'label': labels, 'posting_id': posting_ids }

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item: int) -> dict:
        img = Image.open(os.path.join(
            self.root,
            self.files[item]))
        if not self.is_training:
            img = self.tform(img)
        return {
            'image': img,
            'text': self.texts[item],
            'posting_id': self.posting_ids[item],
            'label': self.labels[item]
        }


class InstanceAugmentationDMLDataset(data.Dataset):
    """
    Dataset used for unsupervised pre-training/auxiliary loss.
    Assigns a class per image and relies on instance augmentation
    to make things work.
    """
    def __init__(self, image_path: str, image_size: int = 224, is_training: bool = True):
        super().__init__()
        self.image_path = image_path
        self.is_training = is_training
        self.image_size = image_size

        self.files = list(glob.glob(self.image_path + '/*.jpg'))
        self.num_classes = len(self.files)
        self.labels = np.arange(self.num_classes) # needed for BalancedBatchSampler

        normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        self.tform = transforms.Compose([
            transforms.RandomResizedCrop(self.image_size, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            #transforms.RandomApply([moco.loader.GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, item: int) -> dict:
        img = Image.open(os.path.join(
            self.image_path, self.files[item]))
        return {
            'image': img,
            'label': item,
            'text': '',
            'posting_id': -1
        }

    def collate_fn(self, elements: list) -> dict:
        """
        Collates (pre-processes) a list of elements to a batch.

        Arguments:
            :elements: is a list of dicts which keys image, text and label.
        Returns:
            A dict with keys image text and label containing the batched/preprocessed
            versions of the text, label and image.
        """
        images1 = torch.zeros((len(elements), 3, self.image_size, self.image_size), dtype=torch.float32)
        images2 = torch.zeros((len(elements), 3, self.image_size, self.image_size), dtype=torch.float32)
        labels = torch.zeros((len(elements),), dtype=torch.int64)
        text = []
        posting_ids = []

        for i in range(len(elements)):
            images1[i, ...] = self.tform(elements[i]['image'])
            images2[i, ...] = self.tform(elements[i]['image'])
            text.append(elements[i]['text'])
            posting_ids.append(elements[i]['posting_id'])
            labels[i] = int(elements[i]['label'])

        return {
            'image1': images1,
            'image2': images2,
            'text': text,
            'label': labels,
            'posting_id': posting_ids
        }


class BalancedBatchSampler(torch.utils.data.sampler.Sampler):
    """
    Samples a balanced batch of a pre-defined number of classes
    and a fixed number of images per class.
    """
    def __init__(self, dataset: DMLDataset, batch_size: int, samples_per_class: int=2):
        super().__init__(dataset)
        self.batch_size = batch_size
        self.samples_per_class = samples_per_class
        self.dataset = dataset

        self.class_list = np.arange(self.dataset.num_classes)
        np.random.shuffle(self.class_list)

        self.class_to_indices = dict()
        for c in self.class_list:
            self.class_to_indices[c] = np.nonzero(self.dataset.labels == c)[0]

    def __iter__(self):
        np.random.shuffle(self.class_list)
        for i in self.class_list:
            idxs = self.class_to_indices[i]
            try:
                sample_idxs = np.random.choice(idxs, size=self.samples_per_class, replace=False)
            except ValueError as e:
                sample_idxs = np.random.choice(idxs, size=self.samples_per_class, replace=True)
            for ix in sample_idxs:
                yield ix

    def __len__(self):
        return len(self.class_list) * self.samples_per_class


def get_val_labels(path: str, train_labels: set) -> set:
    df = pd.read_csv(path)
    all_labels = set(df['label_group'].unique())
    return all_labels - train_labels
