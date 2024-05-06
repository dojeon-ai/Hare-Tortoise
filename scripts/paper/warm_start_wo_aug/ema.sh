#!/bin/bash
cd ../../../

ema_rates=(0.999)
configs=("cifar10_resnet18" "cifar100_vit-ti" "tig_vgg16")

for seed in {1..5}; do
    for config in "${configs[@]}"; do
        for ema in "${ema_rates[@]}"; do
            python run.py \
                --config_name "w_$config"\
                --overrides group_name=warm_start \
                --overrides exp_name="$config"_ema"$ema" \
                --overrides trainer.ema="$ema" \
                --overrides seed="$seed"
        done
    done
done