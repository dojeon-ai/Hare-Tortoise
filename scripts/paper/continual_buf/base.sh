#!/bin/bash
cd ../../../

# Define the input and label ratios
configs=("cifar10_resnet18" "cifar100_vit-ti" "tig_vgg16")

for seed in {1..5}; do
    for config in "${configs[@]}"; do
        python run.py \
            --config_name "c_$config"\
            --overrides group_name=test_continual_buf \
            --overrides exp_name="$config"_aug_base \
            --overrides dataset.buffer_size="5000" \
            --overrides seed="$seed" 
    done
done