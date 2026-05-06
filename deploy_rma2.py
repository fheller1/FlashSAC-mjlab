import os

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"

import argparse
import sys
from typing import Any

import hydra
import torch
from omegaconf import OmegaConf

from flash_rl.agents.flashSAC.network import FlashSACActor, ProprioAdaptTConv, RunningMeanStd


TASK_ID = "Isaac-Inhand-Rotate-Deploy-Sharpa-Wave-v0"


def deploy(args: argparse.Namespace) -> None:
    OmegaConf.register_new_resolver("eval", lambda s: eval(s))
    hydra.initialize(version_base=None, config_path=args.config_path)
    cfg = hydra.compose(config_name=args.config_name, overrides=args.overrides)
    OmegaConf.resolve(cfg)

    device = torch.device(args.device or "cuda:0")
    device_str = str(device)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # --- deploy env (real robot) ---
    import gymnasium as gym
    import rl_isaaclab.tasks.inhand_rotate  # noqa: F401 — registers the task

    from rl_isaaclab.tasks.inhand_rotate.sharpa_wave_deploy_env_cfg import SharpaWaveEnvCfg
    from rl_isaaclab.wrapper.sharpa_wave_deploy_env_wrapper import GymStyleEnvWrapper

    env_cfg = SharpaWaveEnvCfg()
    env_cfg.device = device_str
    env_cfg.seed = args.seed
    if args.hand_side is not None:
        env_cfg.hand_side = args.hand_side
    if args.enable_on_board:
        env_cfg.enable_on_board = True
    if args.grasp_cache_path:
        env_cfg.grasp_cache_path = args.grasp_cache_path

    env = gym.make(TASK_ID, cfg=env_cfg, render_mode=None)
    env = GymStyleEnvWrapper(env, clip_actions=env_cfg.clip_actions)

    # --- Stage 1 actor (env_mlp bypassed) ---
    priv_info_dim: int = int(cfg.agent.env_mlp_units[-1]) if cfg.agent.env_mlp_units else 8
    obs_size = env_cfg.observation_space  # 192
    action_dim = env_cfg.action_space      # 22

    actor_net = FlashSACActor(
        num_blocks=cfg.agent.actor_num_blocks,
        input_dim=obs_size + priv_info_dim,  # 200
        hidden_dim=cfg.agent.actor_hidden_dim,
        action_dim=action_dim,
        priv_info_dim=0,   # _encode_priv is a no-op; e is supplied externally
        env_mlp=None,
    ).to(device)

    ckpt = torch.load(os.path.join(args.stage1_checkpoint_path, "actor.pt"), map_location=device)
    state_dict = ckpt["network_state_dict"]
    if any(k.startswith("_orig_mod.") for k in state_dict):
        state_dict = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
    model_keys = set(actor_net.state_dict().keys())
    state_dict = {k: v for k, v in state_dict.items() if k in model_keys}
    actor_net.load_state_dict(state_dict)
    actor_net.eval()
    for p in actor_net.parameters():
        p.requires_grad_(False)
    print(f"\033[32m[RMA2]\033[0m Loaded Stage 1 actor from {args.stage1_checkpoint_path}")

    # --- Stage 2 adapt_tconv + sa_mean_std ---
    frame_dim = obs_size // 3  # 64
    adapt_tconv = ProprioAdaptTConv(frame_dim=frame_dim, latent_dim=priv_info_dim).to(device)
    sa_mean_std = RunningMeanStd((env_cfg.prop_hist_len, frame_dim)).to(device)

    adapt_ckpt = torch.load(args.adapt_checkpoint_path, map_location=device)
    adapt_tconv.load_state_dict(adapt_ckpt["adapt_tconv"])
    sa_mean_std.load_state_dict(adapt_ckpt["sa_mean_std"])
    adapt_tconv.eval()
    sa_mean_std.eval()
    print(f"\033[32m[RMA2]\033[0m Loaded adapt_tconv from {args.adapt_checkpoint_path}")

    # --- deployment loop ---
    obs_dict = env.reset()
    print("\033[32m[RMA2]\033[0m Running. Press Ctrl+C to stop.")

    try:
        while True:
            obs = obs_dict["obs"].to(device)                          # (1, 192)
            proprio_hist = obs_dict["proprio_hist"].to(device)        # (1, 30, 64)

            with torch.no_grad():
                ph_norm = sa_mean_std(proprio_hist)                   # normalize, no stat update (eval mode)
                e = torch.tanh(adapt_tconv(ph_norm))                  # (1, 8)
                augmented_obs = torch.cat([obs, e], dim=-1)           # (1, 200)
                mean, _ = actor_net.get_mean_and_std(augmented_obs, training=False)
                actions = torch.tanh(mean)                            # (1, 22) in [-1, 1]

            obs_dict, _reward, _done, _info = env.step(actions)      # env enforces 20Hz + scales actions
    except KeyboardInterrupt:
        print("\n\033[32m[RMA2]\033[0m Stopped.")
        sys.exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deploy FlashSAC RMA Stage 2 on real Sharpa Wave hand")
    parser.add_argument("--config_path", type=str, default="./configs")
    parser.add_argument("--config_name", type=str, default="rma2_base")
    parser.add_argument("--overrides", action="append", default=[])
    parser.add_argument("--stage1_checkpoint_path", type=str, required=True, help="Stage 1 checkpoint dir")
    parser.add_argument("--adapt_checkpoint_path", type=str, required=True, help="Stage 2 .pt file")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hand_side", type=int, default=None, help="0=left, 1=right")
    parser.add_argument("--enable_on_board", action="store_true", help="Use on-board tactile inference")
    parser.add_argument("--grasp_cache_path", type=str, default=None)
    args = parser.parse_args()
    deploy(args)
