import util
config = util.load_config('configs/baseline.py')

config['model']['type'] = 'model.AttentionEnsembleExtractor'
config['is_ensemble'] = True
config['image_size'] = 300
