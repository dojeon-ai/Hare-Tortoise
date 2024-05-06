import os
from PIL import Image
import torchvision
import argparse

# Function to create directory structure
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# Function to convert image to black and white
def convert_to_bw(image):
    return image.convert('1')  # '1' mode is for binary (black and white)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process root directory.')
    parser.add_argument('--root', type=str, default='/home/hojoonlee/assets/MNIST/', help='Root directory path')
    args = parser.parse_args()
    
    root_dir = args.root
    train_dir = os.path.join(root_dir, 'train')
    val_dir = os.path.join(root_dir, 'val')

    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        # Download MNIST dataset
        dataset_train = torchvision.datasets.MNIST(root=root_dir, train=True, download=True)
        dataset_val = torchvision.datasets.MNIST(root=root_dir, train=False, download=True)

        # Define the root directory for train and validation sets
        create_dir(train_dir)
        create_dir(val_dir)

        # Initialize counters for each class (digit)
        train_counters = {str(digit): 0 for digit in range(10)}
        val_counters = {str(digit): 0 for digit in range(10)}

        # Process and save the training images
        for image, label in dataset_train:
            class_name = str(label)
            folder_path = os.path.join(train_dir, class_name)
            create_dir(folder_path)

            bw_image = convert_to_bw(image)  # Convert to black and white

            train_counters[class_name] += 1
            file_name = os.path.join(folder_path, f'{train_counters[class_name]:04d}.png')
            bw_image.save(file_name, 'PNG')

        # Process and save the validation images
        for image, label in dataset_val:
            class_name = str(label)
            folder_path = os.path.join(val_dir, class_name)
            create_dir(folder_path)

            bw_image = convert_to_bw(image)  # Convert to black and white

            val_counters[class_name] += 1
            file_name = os.path.join(folder_path, f'{val_counters[class_name]:04d}.png')
            bw_image.save(file_name, 'PNG')

