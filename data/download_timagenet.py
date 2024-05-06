import os
import requests
from PIL import Image
from io import BytesIO
import zipfile
import shutil
import argparse

# Function to create directory structure
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process root directory.')
    parser.add_argument('--root', type=str, default='/home/hojoonlee/assets/TIG/', help='Root directory path')
    args = parser.parse_args()
    
    root_dir = args.root
    train_dir = os.path.join(root_dir, 'train')
    val_dir = os.path.join(root_dir, 'val')

    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        # Download TIG dataset
        response = requests.get("http://cs231n.stanford.edu/tiny-imagenet-200.zip")
        if response.status_code == 200:
            with zipfile.ZipFile(BytesIO(response.content)) as zip_ref:
                zip_ref.extractall(root_dir)
        else:
            raise Exception(f"Failed to download dataset, status code: {response.status_code}")
            
        # Define root directory, train and validation directories
        create_dir(train_dir)
        create_dir(val_dir)

        # Move train data to train_dir
        os.rename(root_dir + '/tiny-imagenet-200/train', train_dir)

        # Separating validation images into separate sub folders
        f = open(root_dir + '/tiny-imagenet-200/val/val_annotations.txt', 'r')
        data = f.readlines()
        val_img_dict = {}
        for line in data:
            words = line.split('\t')
            val_img_dict[words[0]] = words[1]
        f.close()

        val_counters = {label: 0 for label in set(val_img_dict.values())}
        for image, label in val_img_dict.items():
            folder_path = os.path.join(val_dir, label)
            create_dir(folder_path)

            val_counters[label] += 1
            os.rename(os.path.join(root_dir + '/tiny-imagenet-200/val/images/', image), os.path.join(folder_path, label+f'_{val_counters[label]}.JPEG'))
