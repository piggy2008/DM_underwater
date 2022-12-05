import os
from PIL import Image
path = '/home/ty/Downloads/clear'
gt_path = ''
output_path = 'dataset/water_train_16_128'
images = os.listdir(path)
if not os.path.exists(os.path.join(output_path, 'hr_128')):
    os.makedirs(os.path.join(output_path, 'hr_128'))


for i, image_name in enumerate(images):
    image = Image.open(os.path.join(path, image_name)).convert("RGB")
    image = image.resize((128, 128))
    image.save(
        '{}/hr_{}/{}.png'.format(output_path, '128', str(i).zfill(5)))
