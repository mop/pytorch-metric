import util
config = util.load_config('configs/baseline.py')

config['model']['type'] = 'model.EnsembleExtractor'
config['is_ensemble'] = True
