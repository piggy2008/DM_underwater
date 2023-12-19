import os
from PIL import Image
path = '/home/ty/data/UIEB/input_test_uw'
gt_path = ''
output_path = 'dataset/uieb_val_16_256'
images = os.listdir(path)
if not os.path.exists(os.path.join(output_path, 'sr_16_256')):
    os.makedirs(os.path.join(output_path, 'sr_16_256'))


for i, image_name in enumerate(images):
    image = Image.open(os.path.join(path, image_name)).convert("RGB")
    image = image.resize((256, 256))
    image.save(
        '{}/sr_16_{}/{}.png'.format(output_path, '256', str(i).zfill(5)))
