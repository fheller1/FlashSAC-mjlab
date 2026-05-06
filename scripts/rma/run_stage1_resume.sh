#!/bin/bash
##################################################################################
# FlashSAC RMA Stage 1 — Resume from 50M checkpoint, train for 150M total steps
##################################################################################

uv run --frozen python train.py \
    --config_name flashSAC_base \
    --overrides env=sharpa \
    --overrides num_env_steps=150_002_688 \
    --overrides num_train_envs=1024 \
    --overrides num_eval_envs=null \
    --overrides num_record_envs=null \
    --overrides num_eval_episodes=1024 \
    --overrides num_record_episodes=0 \
    --overrides agent=flashSAC_rma \
    --overrides agent.buffer_max_length=10_000_000 \
    --overrides agent.buffer_min_length=100_000 \
    --overrides updates_per_interaction_step=2 \
    --overrides n_step=3 \
    --overrides agent_load_path=models/sharpa-benchmark/priv-info/Isaac-Inhand-Rotate-Sharpa-Wave-v0/seed0-0429-071433/step48829 \
    --overrides env.device=cuda:0 \
    --overrides agent.device_type=cuda:0
