import numpy as np
import argparse
import text_utils
import data
import torch.utils.data as data_util
import logging
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import pairwise_distances
import pickle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--log-filename', default='text-log.txt')
    parser.add_argument('--dataset', default='/home/nax/Projects/shopee-dataset/train.csv')
    parser.add_argument('--label-split', default='/home/nax/Projects/shopee-dataset/train_labels.csv')
    parser.add_argument('--image-size', default=224)
    parser.add_argument('--batch-size', default=32)

    args = parser.parse_args()

    train_labels = np.loadtxt(args.label_split, dtype=np.int64)
    dataset = data.DMLDataset(args.dataset, image_size=args.image_size, subset_labels=train_labels)
    loader = data_util.DataLoader(
            dataset,
            batch_size=args.batch_size,
            collate_fn=dataset.collate_fn)

    val_labels = data.get_val_labels(args.dataset, set(train_labels))
    val_labels = list(val_labels)
    val_dataset = data.DMLDataset(
            args.dataset, 
            image_size=args.image_size,
            is_training=False,
            subset_labels=val_labels)
    val_loader = data_util.DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            collate_fn=val_dataset.collate_fn
    )

    logging.basicConfig(
        format='%(asctime)s %(message)s',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(f'log/{args.log_filename}'),
            logging.StreamHandler()
        ]
    )

    if False:
        text_model = text_utils.train_text_embeddings(loader)
        text_model.save(args.output)

        val_embedding, all_labels, _ = text_utils.encode_text(text_model, val_loader)
        D = cdist(val_embedding, val_embedding) + 10000 * np.eye(val_embedding.shape[0])
        preds = D.argmin(axis=1)
        preds = all_labels[preds]
        logging.info(f'Accuracy {np.mean(preds.ravel() == all_labels.ravel())}')
    else:
        text_model = text_utils.train_text_tfidf(loader)
        val_embedding, all_labels, _ = text_utils.encode_text_tfidf(text_model, val_loader, max_features=None, ngram_range=(1,2), preprocess_args={'stem': False, 'stopwords': []})
        D = pairwise_distances(X=val_embedding, Y=val_embedding) + 1000 * np.eye(val_embedding.shape[0])
        #D = cdist(val_embedding, val_embedding) + 10000 * np.eye(val_embedding.shape[0])
        preds = D.argmin(axis=1)
        preds = all_labels[preds]
        logging.info(f'Accuracy {np.mean(preds.ravel() == all_labels.ravel())}')

        with open(args.output, 'wb') as fp:
            pickle.dump(val_embedding, fp)


if __name__ == "__main__":
    main()
