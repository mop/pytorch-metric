import util
config = util.load_config('configs/baseline.py')

config['model']['type'] = 'model.AttentionEnsembleResNext50Extractor'
config['is_ensemble'] = True
config['image_size'] = 400
