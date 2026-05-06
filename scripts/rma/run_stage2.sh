#!/bin/bash
##################################################################################
# FlashSAC RMA Stage 2 — ProprioAdaptTConv training
# Loads Stage 1 checkpoint: seed0-0429-164347/step97658 (scale 0.6 + random quat)
##################################################################################

uv run --frozen python train_rma2.py \
    --config_name rma2_base \
    --overrides env=sharpa \
    --overrides num_train_envs=1024 \
    --overrides env.device=cuda:0 \
    --checkpoint_path models/sharpa-benchmark/priv-info/Isaac-Inhand-Rotate-Sharpa-Wave-v0/seed0-0429-164347/step97658 \
    --device cuda:0
