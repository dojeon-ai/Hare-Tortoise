#!/bin/bash

configs=("cifar10_resnet18" "cifar100_vit-ti" "tig_vgg16")

for seed in {1..1}; do
    for config in "${configs[@]}"; do
        python run.py \
            --config_name "c_$config"\
            --overrides group_name=test_continual \
            --overrides exp_name="$config"_base \
            --overrides seed="$seed"
    done
done

ema_rates=(0.999)
for seed in {1..1}; do
    for config in "${configs[@]}"; do
        for ema in "${ema_rates[@]}"; do
            python run.py \
                --config_name "c_$config"\
                --overrides group_name=test_continual \
                --overrides exp_name="$config"_hare_tortoise"$ema" \
                --overrides trainer.ema="$ema" \
                --overrides trainer.hare_tortoise.reset_every="10" \
                --overrides seed="$seed"
        done
    done
done


for seed in {1..1}; do
    for config in "${configs[@]}"; do
        python run.py \
            --config_name "c_$config"\
            --overrides group_name=test_continual \
            --overrides exp_name="$config"_head_reset \
            --overrides trainer.reinit.b_lmbda=0.0 \
            --overrides trainer.reinit.h_lmbda=1.0 \
            --overrides seed="$seed"
    done
done

sp_rates=(0.8)
for seed in {1..1}; do
    for config in "${configs[@]}"; do
        for sp in "${sp_rates[@]}"; do
            python run.py \
                --config_name "c_$config"\
                --overrides group_name=test_continual \
                --overrides exp_name="$config"_shrink_perturb"$sp" \
                --overrides trainer.reinit.b_lmbda="$sp" \
                --overrides trainer.reinit.h_lmbda="$sp" \
                --overrides seed="$seed"
        done
    done
done

