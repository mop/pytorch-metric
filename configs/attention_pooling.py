import util
config = util.load_config('configs/baseline.py')

config['model']['type'] = 'model.AttentionExtractor'
