import torch
import torch.nn.functional as F
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
import os
import model_loader
import torch.utils.tensorboard as tensorboard
from torch.autograd import grad
from model import SimSiamEmbeddingPredictor, EmbeddingPredictor, SimSiamEmbedding, SwitchableBatchNorm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='/home/ubuntu/shopee-dataset/train.csv')
    parser.add_argument('--instance-dataset', default='/home/ubuntu/fashion/fashion-dataset/images/')
    parser.add_argument('--label-split', default='/home/ubuntu/shopee-dataset/train_labels.csv')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--embedding-size', type=int)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--warmup_k', type=int)
    parser.add_argument('--image-size', type=int)
    parser.add_argument('--freeze-batchnorm', action='store_true')
    parser.add_argument('--samples-per-class', type=int, default=2)
    parser.add_argument('--apex', action='store_true')
    parser.add_argument('--lr-steps', nargs='+', type=int)
    parser.add_argument('--mode', default='train', choices=('train', 'trainval', 'test'))
    parser.add_argument('--log-filename', default='example')
    parser.add_argument('--config', default='configs/baseline.py')
    parser.add_argument('--output', default='experiments/baseline')
    parser.add_argument('--instance-augmentation-weight', type=float, default=1.0)
    parser.add_argument('--instance-augmentation', action='store_true')

    args = parser.parse_args()

    if args.apex:
        from apex import amp

    config = util.load_config(args.config)
    util.update_args(args, config, additional_keys=('epochs',))

    if args.warmup_k is None:
        args.warmup_k = config['warmup_k']

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    train_labels = np.loadtxt(args.label_split, dtype=np.int64)
    dataset = data.DMLDataset(
            args.dataset,
            image_size=args.image_size,
            mixup_alpha=config['mixup_alpha'],
            subset_labels=train_labels)
    sampler = data.BalancedBatchSampler(
            batch_size=args.batch_size,
            dataset=dataset,
            samples_per_class=args.samples_per_class)
    loader = data_util.DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=8,
            collate_fn=dataset.collate_fn)

    instance_loader = None
    if args.instance_augmentation:
        instance_dataset = data.InstanceAugmentationDMLDataset(
            args.instance_dataset,
            image_size=args.image_size)

        instance_loader = data_util.DataLoader(
            instance_dataset,
            batch_size=args.batch_size,
            drop_last=True,
            num_workers=8,
            collate_fn=instance_dataset.collate_fn)


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
            collate_fn=val_dataset.collate_fn,
            num_workers=4
    )

    print(f'# samples {len(dataset)}')

    backbone = util.get_class_fn(config['model'])()
    backbone.eval()
    num_learners = 1
    in_sizes = []
    if config['is_ensemble']:
        tmp = backbone(torch.rand(1, 3, args.image_size, args.image_size))
        num_learners = len(tmp)
        for i in range(num_learners):
            in_sizes.append(tmp[i].squeeze().size(0))
        in_size = in_sizes[0]
    else:
        in_size = backbone(torch.rand(1, 3, args.image_size, args.image_size)).squeeze().size(0)
        in_sizes.append(in_size)
        
    backbone.train()

    embeddings = []
    for i in range(num_learners):
        emb = torch.nn.Linear(in_sizes[i], args.embedding_size)
        emb.cuda()
        embeddings.append(emb)

    sim_siam = None
    if args.instance_augmentation:
        sim_siam = SimSiamEmbedding()
        sim_siam.cuda()
        model = SimSiamEmbeddingPredictor(backbone, embeddings, sim_siam)
        model.train()
    else:
        model = EmbeddingPredictor(backbone, embeddings)
        model.train()

    def set_bn_eval(m):
        if m.__class__.__name__.find('BatchNorm') != -1:
            print('set bn eval...')
            m.eval()
    if args.freeze_batchnorm:
        model.apply(set_bn_eval)

    if not args.apex:
        model = torch.nn.DataParallel(model)
    if args.instance_augmentation:
        model = SwitchableBatchNorm.convert_switchable_batchnorm(model, 2)
    model = model.cuda()

    criterion_list = []
    for i in range(num_learners):
        criterion = util.get_class_fn(config['criterion'])(
            embedding_size=args.embedding_size,
            num_classes=dataset.num_classes).cuda()
        criterion_list.append(criterion)

    opt_warmup = util.get_class(config['opt']['type'])([
        {
            **{'params': list(backbone.parameters())
            },
            'lr': 0
        },
        {
            **{'params': sum([list(emb.parameters()) for emb in embeddings], [])
            },
            **config['opt']['args']['embedding']
        },
        {
            **{'params': sum([list(c.parameters()) for c in criterion_list], [])
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
            **{'params': sum([list(emb.parameters()) for emb in embeddings], [])
            },
            **config['opt']['args']['embedding']
        },
        {
            **{'params': sum([list(c.parameters()) for c in criterion_list], [])
            },
            **config['opt']['args']['proxynca']
        },
    ], **config['opt']['args']['base'])

    if args.apex:
        (model, *criterion_list), (opt, opt_warmup) = amp.initialize([model] + criterion_list, [opt, opt_warmup], opt_level='O1')
        model = torch.nn.DataParallel(model)

    scheduler = util.get_class(config['lr_scheduler']['type'])(
        opt, **config['lr_scheduler']['args'])

    if not os.path.exists('log'):
        os.makedirs('log')

    if not os.path.exists(args.output):
        os.makedirs(args.output)

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

    instance_aug_iter = None
    if args.instance_augmentation:
        instange_aug_iter = iter(instance_loader)

    for e in range(args.warmup_k):
        for batch in loader:
            imgs = batch['image']
            text = batch['text']
            labels = batch['label']

            if args.instance_augmentation:
                SwitchableBatchNorm.switch_to(model, 0)

            opt_warmup.zero_grad()
            ms = model(imgs.cuda())

            if args.instance_augmentation:
                ms = ms[:-2]

            loss = 0
            for m, criterion in zip(ms, criterion_list):
                loss += criterion(m, labels.cuda())

            if args.instance_augmentation:
                try:
                    instance_batch = next(instance_aug_iter)
                except:
                    instance_aug_iter = iter(instance_loader)
                    instance_batch = next(instance_aug_iter)

                SwitchableBatchNorm.switch_to(model, 1)
                preds1 = model(instance_batch['image1'])
                preds2 = model(instance_batch['image2'])
                z1, p1 = preds1[-2:]
                z2, p2 = preds2[-2:]

                negcos_loss = (negcos(p1, z2) / 2.0 + negcos(p2, z1) / 2.0) * args.instance_augmentation_weight

                #d_loss1, = grad(negcos_loss, (backbone,))
                #d_loss2, = grad(loss, (backbone,))

                #print(d_loss1, d_loss2)
                loss += negcos_loss
                SwitchableBatchNorm.switch_to(model, 0)

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

    prev_lr = 0
    writer = tensorboard.SummaryWriter(args.output)
    best_acc = 0
    for e in range(args.epochs):
        curr_lr = opt.param_groups[0]['lr']
        print(prev_lr, curr_lr)
        if curr_lr != prev_lr:
            prev_lr = curr_lr
            lr_steps.append(e)

        tic_per_epoch = time.time()
        losses_per_epoch = []
        negcos_loss_per_epoch = []
        grad_norm_negcos = []
        grad_norm_loss = []

        for batch in loader:
            imgs, text, labels = batch['image'], batch['text'], batch['label']

            opt.zero_grad()
            it += 1

            if args.instance_augmentation:
                SwitchableBatchNorm.switch_to(model, 0)

            ms = model(imgs.cuda())

            if args.instance_augmentation:
                ms = ms[:-2]

            loss = 0
            for m, criterion in zip(ms, criterion_list):
                loss += criterion(m, labels.cuda())

            if args.instance_augmentation:
                try:
                    instance_batch = next(instance_aug_iter)
                except:
                    instance_aug_iter = iter(instance_loader)
                    instance_batch = next(instance_aug_iter)

                SwitchableBatchNorm.switch_to(model, 1)
                preds1 = model(instance_batch['image1'])
                preds2 = model(instance_batch['image2'])
                SwitchableBatchNorm.switch_to(model, 0)
                z1, p1 = preds1[-2:]
                z2, p2 = preds2[-2:]

                negcos_loss = (negcos(p1, z2) / 2.0 + negcos(p2, z1) / 2.0) * args.instance_augmentation_weight
                #d_loss1, = grad(negcos_loss, (backbone,))
                #d_loss2, = grad(loss, (backbone,))
                #grad_norm_negcos.append(d_loss1.detach().cpu().numpy())
                #grad_norm_loss.append(d_loss2.detach().cpu().numpy())
                loss += negcos_loss
                negcos_loss_per_epoch.append(negcos_loss.detach().cpu().numpy())

            if args.apex:
                with amp.scale_loss(loss, opt) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            first_param = list(model.parameters())[0]
            #print(first_param.size(), np.linalg.norm(first_param.grad.cpu().numpy()))

            torch.nn.utils.clip_grad_value_(model.parameters(), 10)
            losses_per_epoch.append(loss.data.detach().cpu().numpy())
            #print(losses_per_epoch[-1])

            opt.step()

        toc_per_epoch = time.time()
        print(opt)
        logging.info(f'epoch: {e} in {toc_per_epoch - tic_per_epoch}')

        losses.append(np.mean(losses_per_epoch))

        tic_val = time.time()
        acc = eval_utils.evaluate(model, val_loader)
        toc_val = time.time()

        if args.freeze_batchnorm:
            model.apply(set_bn_eval)

        if acc > best_acc:
            logging.info('found new best accuracy, saving model...')
            best_acc = acc
            torch.save({
                'epoch': e,
                'state_dict': model.state_dict(), 
                'accuracy': best_acc,
                'optimizer': opt.state_dict(),
                'amp': amp.state_dict() if args.apex else None
            }, os.path.join(args.output, f'model_best_epoch_{e}.pth'))

        scores.append(acc)
        scheduler.step(acc)
        
        logging.info(f'Accuracy: {acc} in epoch: {e}, loss: {losses[-1]}, val_time: {toc_val - tic_val}, negcos: {np.mean(negcos_loss_per_epoch)}')
        #logging.info(f'grad_negcos: {np.mean(grad_norm_negcos)} grad_loss: {np.mean(grad_norm_loss)}')
        writer.add_scalar('loss_train', losses[-1], e)
        writer.add_scalar('val_accuracy', scores[-1], e)
        writer.add_scalar('train_time', toc_per_epoch - tic_per_epoch, e)
        writer.add_scalar('val_time', toc_val - tic_val, e)
        writer.add_scalar('learning_rate', lr_steps[-1], e)

        writer.flush()
    writer.close()

        # step the scheduler if accuracy does not increase.
        scheduler.step(acc)


def negcos(p, z):
    # z = z.detach()
    p = F.normalize(p, dim=1)
    z = F.normalize(z, dim=1)
    return -(p*z.detach()).sum(dim=1).mean()


    
if __name__ == '__main__':
    main()
