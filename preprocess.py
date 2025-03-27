from tqdm import tqdm
import numpy as np
import cv2
import os

print('Start')
size = 128
exc_counter = 0
ds_path = 'D:\Documentation_Python\Pet-projects\CatDogCLS\dataset'
f_path = f'D:\Documentation_Python\Pet-projects\CatDogCLS\dataset{size}'

def resize(img_path, size):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (size, size))
    return img

for part in ['train', 'valid', 'test']:
    for cls in ['Cat', 'Dog']:
        os.makedirs(os.path.join(f_path, part, cls), exist_ok=True)
        cls_path = os.path.join(ds_path, part, cls)
        img_list = os.listdir(cls_path)
        print(cls_path, len(img_list))
        for image in tqdm(img_list):
            img_path = os.path.join(cls_path, image)
            try:
                img = resize(img_path, size)
                filename = os.path.join(f_path, part, cls, image)
                cv2.imwrite(filename, img)
            except Exception as e:
                #print(img_path, e)
                exc_counter += 1

print('exc_counter', exc_counter)
print('Done!')
