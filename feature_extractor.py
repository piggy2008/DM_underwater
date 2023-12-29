import numpy as np
import torch
import torchvision
import torchextractor as tx
from torchvision import transforms
import cv2
import os
import json

transform = transforms.Compose([
  transforms.ToPILImage(),
  # transforms.CenterCrop(512),
  transforms.Resize(256),
  transforms.ToTensor()
])

model = torchvision.models.resnet50(pretrained=True).cuda()
# print(tx.list_module_names(model))
model = tx.Extractor(model, ['fc'])


path = '/home/ty/data/val'
features_output = {}
images = os.listdir(path)
images.sort()
for image_name in images:
    image = cv2.imread(os.path.join(path, image_name))
    image = transform(image).unsqueeze(0).cuda()
    print(image_name)
    # dummy_input = torch.rand(7, 3, 224, 224)
    model_output, features = model(image)
    features_output[image_name] = features['fc'].detach().cpu().numpy()
    # print(features_output)
np.save('feature_imagenet.npy', features_output)
# json = json.dumps(features_output)
# f = open('feature_underwater.json', 'w')
# f.write(json)
# f.close()