#!/bin/bash
cd ../../../

configs=("cifar10_resnet18" "cifar100_vit-ti" "tig_vgg16")
 
for seed in {1..5}; do
    for config in "${configs[@]}"; do
        python run.py \
            --config_name "w_$config"\
            --overrides group_name=warm_start \
            --overrides exp_name="$config"_crelu \
            --overrides model.head.activ_type="crelu" \
            --overrides seed="$seed"
    done
done