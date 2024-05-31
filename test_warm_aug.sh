#!/bin/bash

for seed in {1..1}; do
    for config in "${configs[@]}"; do
        python run.py \
            --config_name "wa_$config"\
            --overrides group_name=test_warm_start_aug \
            --overrides exp_name="$config"_base \
            --overrides seed="$seed"
    done
done

ema_rates=(0.999)
configs=("cifar10_resnet18" "cifar100_vit-ti" "tig_vgg16")

for seed in {1..1}; do
    for config in "${configs[@]}"; do
        for ema in "${ema_rates[@]}"; do
            python run.py \
                --config_name "wa_$config"\
                --overrides group_name=test_warm_start_aug \
                --overrides exp_name="$config"_ema"$ema" \
                --overrides trainer.ema="$ema" \
                --overrides seed="$seed"
        done
    done
done

ema_rates=(0.999)
configs=("cifar10_resnet18" "cifar100_vit-ti" "tig_vgg16")

for seed in {1..1}; do
    for config in "${configs[@]}"; do
        for ema in "${ema_rates[@]}"; do
            python run.py \
                --config_name "wa_$config"\
                --overrides group_name=test_warm_start_aug \
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
            --config_name "wa_$config"\
            --overrides group_name=test_warm_start_aug \
            --overrides exp_name="$config"_head_reset \
            --overrides trainer.reinit.b_lmbda=0.0 \
            --overrides trainer.reinit.h_lmbda=1.0 \
            --overrides seed="$seed"
    done
done

weight_decay=(0.1)
for seed in {1..1}; do
    for config in "${configs[@]}"; do
        for wd in "${weight_decay[@]}"; do
            python run.py \
                --config_name "wa_$config"\
                --overrides group_name=test_warm_start_aug \
                --overrides exp_name="$config"_wd"$wd" \
                --overrides trainer.optimizer.weight_decay="$wd" \
                --overrides seed="$seed"
        done
    done
done


ema_rates=(0.999)
configs=("cifar10_resnet18" "cifar100_vit-ti" "tig_vgg16")

for seed in {1..1}; do
    for config in "${configs[@]}"; do
        for ema in "${ema_rates[@]}"; do
            python run.py \
                --config_name "wa_$config"\
                --overrides group_name=test_warm_start_aug \
                --overrides exp_name="$config"_self_distill"$ema" \
                --overrides trainer.ema="$ema" \
                --overrides trainer.self_distill.lmbda="1.0" \
                --overrides seed="$seed"
        done
    done
done

sp_rates=(0.8)
configs=("cifar10_resnet18" "cifar100_vit-ti" "tig_vgg16")

for seed in {1..1}; do
    for config in "${configs[@]}"; do
        for sp in "${sp_rates[@]}"; do
            python run.py \
                --config_name "wa_$config"\
                --overrides group_name=warm_start_aug \
                --overrides exp_name="$config"_shrink_perturb"$sp" \
                --overrides trainer.reinit.b_lmbda="$sp" \
                --overrides trainer.reinit.h_lmbda="$sp" \
                --overrides seed="$seed"
        done
    done
done

