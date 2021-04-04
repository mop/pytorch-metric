import numpy as np
import torch
import argparse
import util
import data
import model
import torch.utils.data as data_util
from scipy.spatial.distance import cdist


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--dataset', default='/home/nax/Downloads/shopee-product-matching/test.csv')
    parser.add_argument('--config', default='configs/baseline.py')
    parser.add_argument('--apex', action='store_true')
    parser.add_argument('--embedding-size', type=int)
    parser.add_argument('--batch-size', type=int)

    args = parser.parse_args()
    threshold = 0.9075778192639249

    config = util.load_config(args.config)

    if args.embedding_size is None:
        args.embedding_size = config['embedding_size']
    if args.batch_size is None:
        args.batch_size = config['batch_size']

    if args.apex:
        from apex import amp

    val_dataset = data.DMLDataset(args.dataset, is_training=False, is_testing=True)
    val_loader = data_util.DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            collate_fn=val_dataset.collate_fn
    )

    backbone = util.get_class_fn(config['model'])()
    backbone.eval()
    in_size = backbone(torch.rand(1, 3, 224, 224)).squeeze().size(0)
    backbone.train()

    emb = torch.nn.Linear(in_size, args.embedding_size)
    model = torch.nn.Sequential(backbone, emb)
    model.eval()

    if not args.apex:
        model = torch.nn.DataParallel(model)
    model = model.cuda()

    if args.apex:
        model = amp.initialize(model, opt_level='O1')
        model = torch.nn.DataParallel(model)

    states = torch.load(args.model)
    model.load_state_dict(states['state_dict'])
    if args.apex:
        amp.load_state_dict(states['amp'])

    model.eval()

    all_fvecs = []
    all_ids = []
    for batch in val_loader:
        all_fvecs.append(model(batch['image'].cuda()).detach().cpu().numpy())
        all_ids += batch['posting_id']
    all_fvecs = np.vstack(all_fvecs)
    all_ids = np.asarray(all_ids)
    D = cdist(all_fvecs, all_fvecs)

    preds = D <= threshold

    for i, p in enumerate(preds):
        print(','.join(list(all_ids[p])))
        #print(list(all_ids[p]))


if __name__ == "__main__":
    main()
