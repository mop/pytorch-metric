import torch
import argparse
import data
import torch.utils.data as data_util
import matplotlib.pyplot as plt
import numpy as np


def main():
    train_labels = np.loadtxt('/home/ubuntu/shopee-dataset/train_labels.csv', dtype=np.int64)
    dataset = data.DMLDataset('/home/ubuntu/shopee-dataset/train.csv', subset_labels=train_labels)
    sampler = data.BalancedBatchSampler(batch_size=32, dataset=dataset, samples_per_class=2)
    loader = data_util.DataLoader(dataset, batch_size=32, sampler=sampler, collate_fn=dataset.collate_fn, num_workers=4)

    print(f'# samples {len(dataset)}')
    for epoch in range(2):
        print(f'Epoch: {epoch}')
        for el in loader:
            print(el['label'].detach().numpy().argmax(axis=1))
            #for i in range(len(el['image'])):
            #    img1 = el['image'][i].detach().numpy()
            #    img1 = np.transpose(img1, (1, 2, 0))

            #    plt.imshow(img1); plt.show()
            #print(el)
        print(f'Epoch: {epoch}')
    
if __name__ == '__main__':
    main()
