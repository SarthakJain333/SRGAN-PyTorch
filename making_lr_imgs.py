import os
import numpy as np
import PIL
from PIL import Image
import torch
from torchvision.transforms import transforms

data_dir = 'celeb_1k'
# os.mkdir('lr_imgs')
lr_imgs = 'lr_imgs'
hr_height=256
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

for img_path in os.listdir(data_dir):
    img = Image.open(data_dir+'/'+img_path)
    lr_transform = transforms.Compose(
            [
                transforms.Resize((hr_height // 4, 
                hr_height // 4), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
    normal_pil_transform = transforms.ToPILImage()
    img = lr_transform(img)
    img = normal_pil_transform(img)
    img.save(lr_imgs+'/'+img_path)
    
print('Done!!')
