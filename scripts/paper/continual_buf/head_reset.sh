#!/bin/bash
cd ../../../

configs=("cifar10_resnet18" "cifar100_vit-ti" "tig_vgg16")

for seed in {1..5}; do
    for config in "${configs[@]}"; do
        python run.py \
            --config_name "c_$config"\
            --overrides group_name=test_continual_buf \
            --overrides exp_name="$config"_head_reset \
            --overrides trainer.reinit.b_lmbda=0.0 \
            --overrides trainer.reinit.h_lmbda=1.0 \
            --overrides dataset.buffer_size="5000" \
            --overrides seed="$seed"
    done
done