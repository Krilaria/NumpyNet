Cats/dogs classification project, made using olny numpy, cv2 and os libraries (not torch, TF, etc.)

dataset_split.py - train/valid/test dataset splitting to folder structure

preprocess.py - resizing all images

dataloader.py - transform images and labels to tensors, resize and normalization

model.py - fully connected and convolutional neural networks, custom architectures

train.py - train and lavid loop, creating loss and accuracy plots

Dataset - https://www.kaggle.com/datasets/bhavikjikadara/dog-and-cat-classification-dataset/data
