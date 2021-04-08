config = {
    'name': 'baseline',
    'criterion': {
        'type': 'loss.ProxyNCA_prob',
        'args': {
            'scale': 3
        }
    },
    'lr_scheduler': {
        'type': 'torch.optim.lr_scheduler.ReduceLROnPlateau',
        'args': {
            'mode': 'max',
            'patience': 4*1
        }
    },
    'lr_scheduler2': {
        'type': 'torch.optim.lr_scheduler.MultiStepLR'
    },

    'epochs': 40*2,
    'batch_size': 128,
    'embedding_size': 2048,
    'image_size': 224,
    'is_ensemble': False,
    'samples_per_class': 2,
    'num_gradcum': 1,
    'mixup_alpha': 0.0,
    'warmup_k': 5,
    'model': {
        'type': 'model.Extractor'
    },
    'opt': {
        'type': 'torch.optim.Adam',
        'args': {
            'backbone': {
                'weight_decay': 0,
            },
            'embedding': {
                'weight_decay': 0,
            },
            'proxynca': {
                'weight_decay': 0,
                'lr': 4e1
            },
            'base': {
                'lr': 4e-3,
                'eps': 1.0
            },
        }
    }
}
