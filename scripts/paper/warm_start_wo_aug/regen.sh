#!/bin/bash
cd ../../../

regen_rates=(1.0 0.1 0.01 0.001 0.0001)
configs=("cifar10_resnet18" "cifar100_vit-ti" "tig_vgg16")
 
for seed in {1..5}; do
    for config in "${configs[@]}"; do
        for regen in "${regen_rates[@]}"; do
            python run.py \
                --config_name "w_$config"\
                --overrides group_name=warm_start \
                --overrides exp_name="$config"_regen"$regen" \
                --overrides trainer.regen.b_lmbda="$regen" \
                --overrides trainer.regen.h_lmbda="$regen" \
                --overrides seed="$seed"
        done
    done
done