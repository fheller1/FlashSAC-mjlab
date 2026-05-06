import os

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"
os.environ["JAX_DEFAULT_MATMUL_PRECISION"] = "highest"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"

import argparse
import random
import sys
from datetime import datetime
from typing import Any, cast

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

from flash_rl.agents import create_agent
from flash_rl.agents.flashSAC.network import ProprioAdaptTConv, RunningMeanStd
from flash_rl.envs.isaaclab import make_isaaclab_env


def train(args: argparse.Namespace) -> None:
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

    device_str = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)

    # Patch agent config for Stage 2: no compilation, minimal buffer, no optimizer restore
    OmegaConf.update(cfg, "agent.device_type", device_str)
    OmegaConf.update(cfg, "agent.use_compile", False)
    OmegaConf.update(cfg, "agent.buffer_max_length", 1000)
    OmegaConf.update(cfg, "agent.buffer_min_length", 1000)
    OmegaConf.update(cfg, "agent.load_optimizer", False)

    # --- env ---
    raw_overrides = getattr(cfg.env, "env_cfg_overrides", None) or {}
    try:
        env_cfg_overrides: dict[str, Any] = OmegaConf.to_container(raw_overrides, resolve=True)  # type: ignore
    except Exception:
        env_cfg_overrides = dict(raw_overrides)

    env = make_isaaclab_env(
        env_name=cfg.env.env_name,
        num_envs=cfg.num_train_envs,
        seed=cfg.seed,
        use_priv_info=True,
        env_cfg_overrides=env_cfg_overrides or None,
        device=device_str,
    )

    # Disable gravity curriculum and set full gravity (same as play/record)
    import carb
    from isaaclab.sim import SimulationContext  # noqa: F401
    unwrapped = cast(Any, env.envs.unwrapped)
    unwrapped.cfg.gravity_curriculum = False
    env.envs.unwrapped.physics_sim_view.set_gravity(carb.Float3(0.0, 0.0, -9.81))

    obs, env_info = env.reset(random_start_init=True)
    current_proprio_hist = env_info.get("proprio_hist")  # (num_envs, 30, 64) or None

    priv_info_dim: int = int(env_info.get("priv_info_dim", 0))
    assert priv_info_dim > 0, "Stage 2 requires priv_info_dim > 0 (set use_priv_info=true)"

    # --- Stage 1 agent (actor frozen, critic/buffer unused) ---
    agent = create_agent(env.observation_space, env.action_space, env_info, cfg.agent)
    agent.load(args.checkpoint_path)
    agent._actor.network.eval()
    for p in agent._actor.network.parameters():
        p.requires_grad_(False)

    # --- ProprioAdaptTConv ---
    frame_dim = env.obs_size[0] // 3  # 192 // 3 = 64
    adapt_tconv = ProprioAdaptTConv(frame_dim=frame_dim, latent_dim=priv_info_dim).to(device)
    adapt_tconv.train()
    optim = torch.optim.Adam(adapt_tconv.parameters(), lr=cfg.adapt_lr)

    # Running mean/std for proprio_hist normalization (trained online, like sharpa)
    sa_mean_std = RunningMeanStd((30, frame_dim)).to(device)
    sa_mean_std.train()

    # --- output directory: sibling of the stage1 checkpoint dir ---
    stage1_run_dir = os.path.normpath(os.path.join(os.path.dirname(args.checkpoint_path), ".."))
    stage2_dir = os.path.join(stage1_run_dir, "stage2")
    os.makedirs(stage2_dir, exist_ok=True)

    # --- WandB ---
    run_name = f"stage2-{datetime.now().strftime('%m%d-%H%M%S')}"
    try:
        import wandb
        wandb.init(
            project=cfg.project_name,
            entity=cfg.entity_name,
            group=getattr(cfg, "group_name", "sharpa-rma2"),
            name=run_name,
            config={
                "checkpoint_path": args.checkpoint_path,
                "num_env_steps": cfg.num_env_steps,
                "num_train_envs": cfg.num_train_envs,
                "adapt_lr": cfg.adapt_lr,
                "frame_dim": frame_dim,
                "latent_dim": priv_info_dim,
                "device": device_str,
            },
        )
    except Exception:
        pass

    # --- training loop ---
    num_envs = env.num_envs
    agent_steps = 0
    last_log_step = 0
    last_save_step = 0
    episode_returns = np.zeros(num_envs, dtype=np.float32)
    recent_returns: list[float] = []
    best_mean_return = -float("inf")
    last_loss = 0.0

    prev_transition: dict[str, Any] = {"next_observation": obs}

    while agent_steps < cfg.num_env_steps:
        # --- adapt loss on current (proprio_hist, priv_info) ---
        if current_proprio_hist is not None:
            ph = torch.as_tensor(current_proprio_hist, dtype=torch.float32, device=device)
            pi = torch.as_tensor(obs[..., -priv_info_dim:], dtype=torch.float32, device=device)

            ph_norm = sa_mean_std(ph)
            e = torch.tanh(adapt_tconv(ph_norm))
            with torch.no_grad():
                e_gt = agent._actor.network.env_mlp(pi)  # already tanh'd by build_env_mlp

            loss = ((e - e_gt.detach()) ** 2).mean()
            optim.zero_grad()
            loss.backward()
            optim.step()
            last_loss = loss.item()

        # --- collect transition from frozen Stage 1 policy ---
        with torch.no_grad():
            actions = agent.sample_actions(agent_steps, prev_transition, training=False)

        next_obs, rewards, terminateds, truncateds, infos = env.step(np.array(actions))
        agent_steps += num_envs

        episode_returns += rewards
        dones = np.logical_or(terminateds, truncateds)
        for i in range(num_envs):
            if dones[i]:
                recent_returns.append(float(episode_returns[i]))
                if len(recent_returns) > 2000:
                    recent_returns.pop(0)
                episode_returns[i] = 0.0

        current_proprio_hist = infos.get("proprio_hist")
        obs = next_obs
        prev_transition = {"next_observation": obs}

        # --- logging ---
        if agent_steps - last_log_step >= cfg.log_interval:
            last_log_step = agent_steps
            mean_ret = float(np.mean(recent_returns)) if recent_returns else 0.0
            print(
                f"Steps: {agent_steps // 1_000_000:04d}M | "
                f"Adapt Loss: {last_loss:.5f} | "
                f"Mean Return: {mean_ret:.2f}"
            )
            try:
                import wandb
                if wandb.run is not None:
                    wandb.log({"adapt_loss": last_loss, "episode_rewards": mean_ret}, step=agent_steps)
            except Exception:
                pass

        # --- checkpointing ---
        if agent_steps - last_save_step >= cfg.save_interval:
            last_save_step = agent_steps
            mean_ret = float(np.mean(recent_returns)) if recent_returns else 0.0
            ckpt = {
                "adapt_tconv": adapt_tconv.state_dict(),
                "sa_mean_std": sa_mean_std.state_dict(),
                "agent_steps": agent_steps,
            }
            step_id = agent_steps // 1_000_000
            torch.save(ckpt, os.path.join(stage2_dir, f"step{step_id}M.pt"))
            torch.save(ckpt, os.path.join(stage2_dir, "last.pt"))
            if mean_ret > best_mean_return:
                best_mean_return = mean_ret
                torch.save(ckpt, os.path.join(stage2_dir, "best.pt"))
            print(f"\033[32m[RMA2]\033[0m Saved checkpoint at {agent_steps // 1_000_000}M steps.")

    # final save
    ckpt = {
        "adapt_tconv": adapt_tconv.state_dict(),
        "sa_mean_std": sa_mean_std.state_dict(),
        "agent_steps": agent_steps,
    }
    torch.save(ckpt, os.path.join(stage2_dir, "final.pt"))
    print(f"\033[32m[RMA2]\033[0m Training complete. Final checkpoint saved to {stage2_dir}/final.pt")

    try:
        import wandb
        if wandb.run is not None:
            wandb.finish()
    except Exception:
        pass

    env.close()
    sys.exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FlashSAC RMA Stage 2: train ProprioAdaptTConv")
    parser.add_argument("--config_path", type=str, default="./configs")
    parser.add_argument("--config_name", type=str, default="rma2_base")
    parser.add_argument("--overrides", action="append", default=[])
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to Stage 1 checkpoint dir")
    parser.add_argument("--device", type=str, default=None, help="e.g. cuda:0")
    args = parser.parse_args()
    train(args)
