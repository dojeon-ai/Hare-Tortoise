#!/bin/bash
cd ../../../

sp_rates=(0.8)
configs=("cifar10_resnet18" "cifar100_vit-ti" "tig_vgg16")
 
for seed in {1..5}; do
    for config in "${configs[@]}"; do
        for sp in "${sp_rates[@]}"; do
            python run.py \
                --config_name "w_$config"\
                --overrides group_name=warm_start \
                --overrides exp_name="$config"_shrink_perturb"$sp" \
                --overrides trainer.reinit.b_lmbda="$sp" \
                --overrides trainer.reinit.h_lmbda="$sp" \
                --overrides seed="$seed"
        done
    done
done