import numpy as np
import torch
import argparse
import util
import data
import model
from model import EmbeddingPredictor
import torch.utils.data as data_util
import eval_utils
import model_loader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--dataset', default='/home/nax/Downloads/shopee-product-matching/train.csv')
    parser.add_argument('--train-label-split', default='/home/nax/Downloads/shopee-product-matching/train_labels.csv')
    parser.add_argument('--config', default='configs/baseline.py')
    parser.add_argument('--apex', action='store_true')
    parser.add_argument('--embedding-size', type=int)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--image-size', type=int)
    parser.add_argument('--use_cuda', action='store_true')

    args = parser.parse_args()

    config = util.load_config(args.config)

    util.update_args(args, config)

    if args.apex:
        from apex import amp

    train_labels = np.loadtxt(args.train_label_split, dtype=np.int64)
    val_labels = data.get_val_labels(args.dataset, set(train_labels))
    val_labels = list(val_labels)
    #val_dataset = data.DMLDataset(args.dataset, 
    #        image_size=args.image_size,
    #        is_training=False,
    #        onehot_labels=False,
    #        subset_labels=val_labels)
    val_dataset = data.DMLDataset(args.dataset, 
            image_size=args.image_size,
            is_training=False,
            onehot_labels=False)
    val_loader = data_util.DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            num_workers=2,
            collate_fn=val_dataset.collate_fn
    )
    backbone, embeddings, model, states = model_loader.load_model(config, args, args.model)

    model.eval()

    if not args.apex:
        model = torch.nn.DataParallel(model)
    model = model.cuda()

    if args.apex:
        model = amp.initialize(model, opt_level='O1')
        model = torch.nn.DataParallel(model)

    if args.apex:
        amp.load_state_dict(states['amp'])

    model.eval()

    print(f'Val accuracy: {eval_utils.evaluate(model, val_loader, use_cuda=args.use_cuda)}')
    print(f'F1: {eval_utils.f1_evaluate(model, val_loader, use_cuda=args.use_cuda)}')
    print(f'F1: {eval_utils.f1_evaluate(model, val_loader, threshold=1.070329, use_cuda=args.use_cuda)}')


if __name__ == "__main__":
    main()
