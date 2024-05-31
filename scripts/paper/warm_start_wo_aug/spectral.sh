#!/bin/bash
cd ../../../

spectral_rates=(1.0 0.1 0.01 0.001 0.0001 0.00001)
configs=("cifar10_resnet18" "cifar100_vit-ti" "tig_vgg16")
 
for seed in {1..5}; do
    for config in "${configs[@]}"; do
        for spectral in "${spectral_rates[@]}"; do
            python run.py \
                --config_name "w_$config"\
                --overrides group_name=warm_start \
                --overrides exp_name="$config"_spectral"$spectral" \
                --overrides trainer.spectral.lmbda="$spectral" \
                --overrides seed="$seed"
        done
    done
done