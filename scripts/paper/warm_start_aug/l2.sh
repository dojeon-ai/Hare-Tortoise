#!/bin/bash
cd ../../../

weight_decay=(0.1)
configs=("cifar10_resnet18" "cifar100_vit-ti" "tig_vgg16")

for seed in {1..5}; do
    for config in "${configs[@]}"; do
        for wd in "${weight_decay[@]}"; do
            python run.py \
                --config_name "wa_$config"\
                --overrides group_name=warm_start_aug \
                --overrides exp_name="$config"_wd"$wd" \
                --overrides trainer.optimizer.weight_decay="$wd" \
                --overrides seed="$seed"
        done
    done
done