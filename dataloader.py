import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import random as r
import cv2
import os

class My_Dataloader:
    def __init__(self, data_dir, batch_size=16, shuffle=True, img_size=128):
        self.data_dir = data_dir
        self.cls_list = os.listdir(data_dir)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.img_size = img_size
        # {'Cat': 0, 'Dog': 1}
        self.cls_to_inx = {cls: idx for idx, cls in enumerate(self.cls_list)}

        self.file_paths = []
        self.labels = []
        for cls in self.cls_list:
            cls_dir = os.path.join(data_dir, cls)
            cls_path_list = os.listdir(cls_dir)
            for image in cls_path_list:
                self.file_paths.append(os.path.join(cls_dir, image))
                self.labels.append(self.cls_to_inx[cls])

        if self.shuffle:
            self._shuffle_data()

    def _shuffle_data(self):
        indexes = list(range(len(self.labels)))
        r.shuffle(indexes)
        self.file_paths = [self.file_paths[i] for i in indexes]
        self.labels = [self.labels[i] for i in indexes]
        
    def leng(self):
        return len(self.file_paths)
        
    def generate_batches(self):
        n_samples = len(self.file_paths)
        for i in tqdm(range(0, n_samples, self.batch_size)):
        #for i in range(0, n_samples, self.batch_size):
            batch_paths = self.file_paths[i:i+self.batch_size]
            batch_labels = self.labels[i:i+self.batch_size]
            batch_images = []
            for image_path in batch_paths:
                img = cv2.imread(image_path)
                if img.shape[:2] != (self.img_size, self.img_size):
                    img = cv2.resize(img, (self.img_size, self.img_size))
                mean = np.array ([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img = img / 255.0
                img = (img - mean) / std
                batch_images.append(img)
            yield np.array(batch_images), np.array(batch_labels)

    def __iter__(self):
        return self.generate_batches()

#train_path = f'D:\Documentation_Python\Pet-projects\CatDogCLS\dataset128\\train'
#train_loader = My_Dataloader(train_path)

def denormalize(img):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    return (img * std + mean).clip(0, 1)

def BCE(output, y):
    elipson = 1e-15
    output_clipped = np.clip(output, elipson, 1 - elipson)
    loss = - (y * np.log(output_clipped) + (1 - y) * np.log(1 - output_clipped)).mean()
    return loss

# Взять один батч
'''if 2 == 3:
    for batch_images, batch_labels in train_loader:
        print('\n', batch_images.shape)
        exit()
else:
    for batch_images, batch_labels in train_loader:
        # Восстановить первое изображение
        img_denorm = denormalize(batch_images[0])

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(img_denorm)
        plt.title("Денормализованное изображение")
        plt.subplot(1, 2, 2)
        plt.imshow(batch_images[0])
        plt.title("Нормализованное изображение")
        plt.show()
        break'''
