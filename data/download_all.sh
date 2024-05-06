#!/bin/bash

# mnist
echo "Downloading mnist..."
python download_mnist.py

# tiny imagenet
echo "Downloading tiny imagenet..."
python download_timagenet.py

# cifar 10
echo "Downloading cifar10..."
python download_cifar10.py

# cifar 100
echo "Downloading cifar100..."
python download_cifar100.py