import os
from PIL import Image
path = '/home/ty/data/LSUI/test_gt'
gt_path = ''
output_path = 'dataset/water_val_16_224'
images = os.listdir(path)
if not os.path.exists(os.path.join(output_path, 'hr_224')):
    os.makedirs(os.path.join(output_path, 'hr_224'))


for i, image_name in enumerate(images):
    image = Image.open(os.path.join(path, image_name)).convert("RGB")
    image = image.resize((224, 224))
    image.save(
        '{}/hr_{}/{}.png'.format(output_path, '224', str(i).zfill(5)))
