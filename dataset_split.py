import os
import shutil
from tqdm import tqdm
from sklearn.model_selection import train_test_split
print('start')
root_dir = 'D:\Documentation_Python\Pet-projects\CatDogCLS\\archive\PetImages'
final_path = 'D:\Documentation_Python\Pet-projects\CatDogCLS\\dataset'
split = (0.7, 0.2, 0.1)

# Функция для разделения данных
def split_dataset(root_dir, final_path, split):
    train_dir = os.path.join(final_path, 'train')
    valid_dir = os.path.join(final_path, 'valid')
    test_dir = os.path.join(final_path, 'test')

    classes = ['Cat', 'Dog']
    for cls in classes:
        print(cls)
        img_paths = [os.path.join(root_dir, cls, fname) for fname in os.listdir(os.path.join(root_dir, cls))]
        train_paths, temp = train_test_split(img_paths, train_size=split[0], random_state=42)
        valid_paths, test_paths = train_test_split(temp, train_size=split[1]/(split[1]+split[2]), random_state=42)
        
        for path in tqdm(train_paths):
            os.makedirs(os.path.join(train_dir, cls), exist_ok=True)
            shutil.copy(path, os.path.join(train_dir, cls, os.path.basename(path)))
        for path in tqdm(valid_paths):
            os.makedirs(os.path.join(valid_dir, cls), exist_ok=True)
            shutil.copy(path, os.path.join(valid_dir, cls, os.path.basename(path)))
        for path in tqdm(test_paths):
            os.makedirs(os.path.join(test_dir, cls), exist_ok=True)
            shutil.copy(path, os.path.join(test_dir, cls, os.path.basename(path)))

split_dataset(root_dir, final_path, split)
print('Done!')
