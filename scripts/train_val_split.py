import pandas as pd
import argparse
import numpy as np
import random


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv-file', default='/home/nax/Downloads/shopee-product-matching/train.csv')
    parser.add_argument('--seed', default=42)
    parser.add_argument('--output', default='/home/nax/Downloads/shopee-product-matching/train_labels.csv')
    parser.add_argument('--train-fraction', type=float, default=0.9)

    args = parser.parse_args()
    df = pd.read_csv(args.csv_file)
    labels = df['label_group'].unique()

    labels = np.asarray(labels)
    rng = np.arange(len(labels))

    random.seed(args.seed)
    np.random.seed(args.seed)

    np.random.shuffle(rng)
    num_train = int(len(rng) * args.train_fraction)

    train_label_ix = rng[:num_train]
    labels = labels[train_label_ix]
    np.savetxt(args.output, labels, fmt='%d')

    
if __name__ == '__main__':
    main()
