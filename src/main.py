import torch
import argparse
import data
import torch.utils.data as data_util
import matplotlib.pyplot as plt
import numpy as np
import argparse
import util
import time
import eval_utils
import logging
import random


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='/home/nax/Downloads/shopee-product-matching/train.csv')
    parser.add_argument('--label-split', default='/home/nax/Downloads/shopee-product-matching/train_labels.csv')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--embedding-size', type=int)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--warmup_k', type=int)
    parser.add_argument('--samples-per-class', type=int, default=2)
    parser.add_argument('--apex', action='store_true')
    parser.add_argument('--lr-steps', nargs='+', type=int)
    parser.add_argument('--mode', default='train', choices=('train', 'trainval', 'test'))
    parser.add_argument('--log-filename', default='example')
    parser.add_argument('--config', default='configs/baseline.py')

    args = parser.parse_args()

    if args.apex:
        from apex import amp

    config = util.load_config(args.config)

    if args.embedding_size is None:
        args.embedding_size = config['embedding_size']
    if args.batch_size is None:
        args.batch_size = config['batch_size']
    if args.epochs is None:
        args.epochs = config['epochs']

    if args.warmup_k is None:
        args.warmup_k = config['warmup_k']

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    train_labels = np.loadtxt(args.label_split, dtype=np.int64)
    dataset = data.DMLDataset(args.dataset, subset_labels=train_labels)
    sampler = data.BalancedBatchSampler(
            batch_size=args.batch_size, 
            dataset=dataset,
            samples_per_class=args.samples_per_class)
    loader = data_util.DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            collate_fn=dataset.collate_fn)

    val_labels = data.get_val_labels(args.dataset, set(train_labels))
    val_dataset = data.DMLDataset(args.dataset, is_training=False, subset_labels=val_labels)

    print(f'# samples {len(dataset)}')

    backbone = util.get_class_fn(config['model'])()
    backbone.eval()
    in_size = backbone(torch.rand(1, 3, 224, 224)).squeeze().size(0)
    backbone.train()

    emb = torch.nn.Linear(in_size, args.embedding_size)
    model = torch.nn.Sequential(backbone, emb)

    if not args.apex:
        model = torch.nn.DataParallel(model)
    #model = model.cuda()

    loss = util.get_class_fn(config['criterion'])(
        embedding_size=args.embedding_size,
        num_classes=dataset.num_classes)#.cuda()

    opt_warmup = util.get_class(config['opt']['type'])([
        {
            **{'params': list(backbone.parameters())
            },
            'lr': 0
        },
        {
            **{'params': list(emb.parameters())
            },
            **config['opt']['args']['embedding']
        },
        {
            **{'params': list(loss.parameters())
            },
            **config['opt']['args']['proxynca']
        },
    ], **config['opt']['args']['base'])

    opt = util.get_class(config['opt']['type'])([
        {
            **{'params': list(backbone.parameters())
            },
            **config['opt']['args']['backbone']
        },
        {
            **{'params': list(emb.parameters())
            },
            **config['opt']['args']['embedding']
        },
        {
            **{'params': list(loss.parameters())
            },
            **config['opt']['args']['proxynca']
        },
    ], **config['opt']['args']['base'])

    if args.apex:
        (model, loss), (opt, opt_warmup) = amp.initialize((model, criterion), (opt, opt_warmup), opt_level='O1')
        model = torch.nn.DataParallel(model)

    scheduler = util.get_class(config['lr_scheduler']['type'])(
        opt, **config['lr_scheduler']['args'])

    logging.basicConfig(
        format='%(asctime)s %(message)s',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(f'log/{args.log_filename}'),
            logging.StreamHandler()
        ]
    )
    logging.info('Training parameters: {}'.format(vars(args)))
    logging.info('Training for {} epochs'.format(args.epochs))

    tic = time.time()

    logging.info(f'warmup for {args.warmup_k} epochs')

    for e in range(args.warmup_k):
        for batch in loader:
            imgs = batch['image']
            text = batch['text']
            labels = batch['label']

            opt_warmup.zero_grad()
            m = model(imgs.cuda())
            loss = criterion(m, labels.cuda())

            if args.apex:
                with amp.scale_loss(loss, opt_warmup) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), 10)
            opt_warmup.step()

        logging.info(f'warm up iteration {e} finished')

    losses = []
    scores = []

    lr_steps = []
    it = 0

    for e in range(args.epochs):
        curr_lr = opt.param_groups[0]['lr']
        print(prev_lr, curr_lr)
        if curr_lr != prev_lr:
            prev_lr = curr_lr
            lr_steps.append(e)

        tic_per_epoch = time.time()
        losses_per_epoch = []
        tnmi = []

        for batch in loader:
            imgs, text, labels = batch['image'], batch['text'], batch['label']

            opt.zero_grad()
            it += 1
            m = model(imgs.cuda())

            loss = criterion(m, labels.cuda())

            if args.apex:
                with amp.scale_loss(loss, opt) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            torch.nn.utils.clip_grad_value_(model.parameters(), 10)
            losses_per_epoch.append(loss.data.detach().cpu().numpy())

            opt.step()
            opt.zero_grad()

        toc_per_epoch = time.time()
        logging.info(f'epoch: {e} in {toc_per_epoch - tic_per_epoch}')

        losses.append(np.mean(losses_per_epoch))
        acc = eval_utils.evaluate(model, val_dataset)
        scores.append(acc)
        logging.info(f'Accuracy: {acc} in epoch: {e}')

        # step the scheduler if accuracy does not increase.
        scheduler.step(acc)


    
if __name__ == '__main__':
    main()
