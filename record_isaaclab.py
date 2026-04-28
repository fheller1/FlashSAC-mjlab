import os

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"

import argparse
import random
from typing import MutableMapping

import hydra
import imageio
import numpy as np
import torch
from omegaconf import OmegaConf

from flash_rl.agents import create_agent
from flash_rl.envs.isaaclab import make_isaaclab_env
from flash_rl.types import Tensor


def record(args: argparse.Namespace) -> None:
    OmegaConf.register_new_resolver("eval", lambda s: eval(s))
    hydra.initialize(version_base=None, config_path=args.config_path)
    cfg = hydra.compose(config_name=args.config_name, overrides=args.overrides)
    OmegaConf.resolve(cfg)

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    env = make_isaaclab_env(
        env_name=cfg.env.env_name,
        num_envs=args.num_envs,
        seed=cfg.seed,
        headless=True,
        use_priv_info=cfg.env.get("use_priv_info", False),
        env_cfg_overrides=dict(cfg.env.env_cfg_overrides) if cfg.env.get("env_cfg_overrides") else None,
        enable_cameras=True,
        render_mode="rgb_array",
    )

    observations, env_info = env.reset(random_start_init=False)
    agent = create_agent(
        observation_space=env.observation_space,
        action_space=env.action_space,
        env_info=env_info,
        cfg=cfg.agent,
    )
    agent.load(args.checkpoint_path)

    frames: list[np.ndarray] = []
    prev_transition: MutableMapping[str, Tensor] = {"next_observation": observations}
    completed_episodes = 0
    episode_returns = np.zeros(args.num_envs)

    while completed_episodes < args.num_episodes:
        frame = env.render()
        if frame is not None:
            # frame shape from IsaacLab: (H, W, C) or (num_envs, H, W, C) — take env 0
            if frame.ndim == 4:
                frame = frame[0]
            frames.append(frame.astype(np.uint8))

        actions = agent.sample_actions(interaction_step=0, prev_transition=prev_transition, training=False)
        next_observations, rewards, terminateds, truncateds, _ = env.step(np.array(actions))

        episode_returns += rewards
        dones = np.logical_or(terminateds, truncateds)
        for idx in range(args.num_envs):
            if dones[idx]:
                completed_episodes += 1
                print(f"Episode {completed_episodes}: return = {episode_returns[idx]:.2f}")
                episode_returns[idx] = 0.0
                if completed_episodes >= args.num_episodes:
                    break

        prev_transition = {"next_observation": next_observations}

    env.close()

    if frames:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        imageio.mimwrite(args.output, frames, fps=args.fps)
        print(f"Saved {len(frames)} frames to {args.output}")
    else:
        print("No frames captured — check that the env supports rgb_array render_mode.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record a trained FlashSAC IsaacLab agent headlessly")
    parser.add_argument("--config_path", type=str, default="./configs")
    parser.add_argument("--config_name", type=str, default="flashSAC_base")
    parser.add_argument("--overrides", action="append", default=[])
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--output", type=str, default="videos/recording.mp4")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--num_episodes", type=int, default=3)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()
    record(args)
