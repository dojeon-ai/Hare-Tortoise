#!/bin/bash
cd ../../../

configs=("cifar10_resnet18" "cifar100_vit-ti" "tig_vgg16")

for seed in {1..5}; do
    for config in "${configs[@]}"; do
        python run.py \
            --config_name "c_$config"\
            --overrides group_name=continual \
            --overrides exp_name="$config"_base \
            --overrides seed="$seed"
    done
done