#!/bin/bash
##################################################################################
# FlashSAC RMA Stage 1 — Object scale (0.6, 0.6, 1.0) + random hand quat
##################################################################################

uv run --frozen python train.py \
    --config_name flashSAC_base \
    --overrides env=sharpa \
    --overrides num_env_steps=100_001_792 \
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
    --overrides env.env_cfg_overrides.reset_random_quat=true \
    --overrides "+env.env_cfg_overrides.object_cfg.spawn.scale=[0.6,0.6,1.0]" \
    --overrides env.device=cuda:6 \
    --overrides agent.device_type=cuda:6
