import util
config = util.load_config('configs/baseline.py')
config['model']['type'] = 'model.AttentionExtractor'
config['image_size'] = 300
config['name'] = 'attention_image_size_300'
