import torch
import numpy as np
import model
from model import EmbeddingPredictor
import torch.nn as nn
import argparse
import util


def load_model(config: dict, args: argparse.Namespace, path: str) -> (nn.Module, nn.Module, nn.Module, dict):
    """
    loads the model from the given path and returns a tuple consisting
    of the backbone, the list of embeddings, the model and a state dictionary. If the
    model was stored in an old format, we try to convert it to the
    newer format.
    """
    states = torch.load(path)

    # construct new style
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

    model = EmbeddingPredictor(backbone, embeddings)
    model.train()

    try:
        model.load_state_dict(states['state_dict'])
    except:
        for key in list(states['state_dict'].keys()):
            new_key = key.replace('module.0.', 'bases.').replace('module.1.', 'embeddings.0.')
            states['state_dict'][new_key] = states['state_dict'].pop(key)
        try:
            model.load_state_dict(states['state_dict'])
        except:
            states = torch.load(path)
            for key in list(states['state_dict'].keys()):
                new_key = key.replace('module.', '')
                states['state_dict'][new_key] = states['state_dict'].pop(key)
            model.load_state_dict(states['state_dict'])
    return backbone, embeddings, model, states
