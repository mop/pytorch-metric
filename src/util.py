import importlib
import copy

def load_config(config: str) -> dict:
    modname = config.replace('/', '.')
    spec = importlib.util.spec_from_file_location(modname, config)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.config

def get_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod

def get_class_fn(param: dict):
    cls = get_class(param['type'])

    args = {}
    if 'args' in param:
        args = param['args']

    def apply_it(*a, **kwargs):
        tmp = copy.deepcopy(args)
        tmp.update(kwargs)
        return cls(*a, **tmp)
    return apply_it

def update_args(args, config: dict, *, additional_keys: list = None):
    """
    Updates the arguments ``args'' obtained from argument parser
    with the config dictionary.
    """
    keys = ['embedding_size', 'batch_size', 'image_size']
    if additional_keys is not None:
        keys += list(additional_keys)
    for k in keys:
        if getattr(args, k) is None:
            setattr(args, k, config[k])
