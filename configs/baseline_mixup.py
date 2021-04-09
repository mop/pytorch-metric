import util
config = util.load_config('configs/baseline.py')
config['image_size'] = 300
config['mixup_alpha'] = 0.1
config['name'] = 'baseline_image_size_300_mixup'
