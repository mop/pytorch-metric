import torch
import csv
import torch.utils.data as data
import pandas as pd
import numpy as np
import os
from PIL import Image
import torchvision.transforms as transforms


class DMLDataset(data.Dataset):
    """
    This class is responsible for loading the data for deep metric learning.
    """
    def __init__(self, 
                 csv_path: str,
                 image_size: int = 224,
                 is_training: bool = True,
                 is_testing: bool = False,
                 subset_labels=None):
        """
        Initializes the DML dataset.

        Arguments:
            csv_path: Path to the training csv file.
            image_size: size of the images (after pre-processing)
            is_training: True if it is training (and augmentation should be performed)
            is_testing: True if it is testing (images will be loaded from the test folder)
            subset_labels: Subset of labels to load for training (if None, all are loaded)
        """
        self.csv_path = csv_path
        self.image_size = image_size
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
            images[i, ...] = self.tform(elements[i]['image'])
            text.append(elements[i]['text'])
            posting_ids.append(elements[i]['posting_id'])
            if not self.is_testing:
                labels[i] = int(elements[i]['label'])
        return { 'image': images, 'text': text, 'label': labels, 'posting_id': posting_ids }

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item: int) -> dict:
        """

        """
        img = Image.open(os.path.join(
            self.root,
            self.files[item]))
        return {
            'image': img,
            'text': self.texts[item],
            'posting_id': self.posting_ids[item],
            'label': self.labels[item]
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
