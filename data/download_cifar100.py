import os
from PIL import Image
import torchvision
import argparse

# Function to create directory structure
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process root directory.')
    parser.add_argument('--root', type=str, default='/home/hojoonlee/assets/CIFAR100/', help='Root directory path')
    args = parser.parse_args()
    
    root_dir = args.root
    train_dir = os.path.join(root_dir, 'train')
    val_dir = os.path.join(root_dir, 'val')

    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        # Download CIFAR100 dataset
        dataset_train = torchvision.datasets.CIFAR100(root=root_dir, train=True, download=True)
        dataset_val = torchvision.datasets.CIFAR100(root=root_dir, train=False, download=True)

        # Define the root directory for train and validation sets
        create_dir(train_dir)
        create_dir(val_dir)

        # Initialize counters for each class
        train_counters = {class_name: 0 for class_name in dataset_train.classes}
        val_counters = {class_name: 0 for class_name in dataset_val.classes}

        # Process and save the training images
        for image, label in dataset_train:
            class_name = dataset_train.classes[label]
            folder_path = os.path.join(train_dir, class_name)
            create_dir(folder_path)

            train_counters[class_name] += 1
            file_name = os.path.join(folder_path, f'{train_counters[class_name]:04d}.png')
            image.save(file_name, 'PNG')  # Save as PNG

        # Process and save the validation images
        for image, label in dataset_val:
            class_name = dataset_val.classes[label]
            folder_path = os.path.join(val_dir, class_name)
            create_dir(folder_path)

            val_counters[class_name] += 1
            file_name = os.path.join(folder_path, f'{val_counters[class_name]:04d}.png')
            image.save(file_name, 'PNG')  # Save as PNG
