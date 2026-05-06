import os

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"

import argparse
import sys
from typing import Any, cast

import hydra
import imageio
import numpy as np
import torch
from omegaconf import OmegaConf

from flash_rl.agents.flashSAC.network import FlashSACActor, ProprioAdaptTConv, RunningMeanStd, build_env_mlp
from flash_rl.envs.isaaclab import make_isaaclab_env


def record(args: argparse.Namespace) -> None:
    OmegaConf.register_new_resolver("eval", lambda s: eval(s))
    hydra.initialize(version_base=None, config_path=args.config_path)
    cfg = hydra.compose(config_name=args.config_name, overrides=args.overrides)
    OmegaConf.resolve(cfg)

    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    device_str = str(device)

    # --- env (no priv_info: Stage 2 runs blind to privileged info) ---
    raw_overrides = getattr(cfg.env, "env_cfg_overrides", None) or {}
    try:
        env_cfg_overrides: dict[str, Any] = OmegaConf.to_container(raw_overrides, resolve=True)  # type: ignore
    except Exception:
        env_cfg_overrides = dict(raw_overrides)

    env = make_isaaclab_env(
        env_name=cfg.env.env_name,
        num_envs=args.num_envs,
        seed=0,
        use_priv_info=False,
        env_cfg_overrides=env_cfg_overrides or None,
        enable_cameras=True,
        render_mode="rgb_array",
        device=device_str,
    )

    import carb
    from isaaclab.sim import SimulationContext
    unwrapped = cast(Any, env.envs.unwrapped)
    unwrapped.cfg.gravity_curriculum = False
    env.envs.unwrapped.physics_sim_view.set_gravity(carb.Float3(0.0, 0.0, -9.81))

    def _set_camera() -> None:
        sim = SimulationContext.instance()
        if sim is not None:
            sim.set_camera_view(eye=args.camera_eye, target=args.camera_target)

    obs, env_info = env.reset(random_start_init=False)
    _set_camera()
    current_proprio_hist = env_info.get("proprio_hist")

    # --- Stage 1 actor (env_mlp bypassed: priv_info_dim=0) ---
    priv_info_dim: int = int(cfg.agent.env_mlp_units[-1]) if cfg.agent.env_mlp_units else 8
    actor_input_dim = env.obs_size[0] + priv_info_dim  # 192 + 8 = 200
    action_dim = env.action_size[0]

    actor_net = FlashSACActor(
        num_blocks=cfg.agent.actor_num_blocks,
        input_dim=actor_input_dim,
        hidden_dim=cfg.agent.actor_hidden_dim,
        action_dim=action_dim,
        priv_info_dim=0,   # _encode_priv is a no-op; we supply e externally
        env_mlp=None,
    ).to(device)

    ckpt = torch.load(os.path.join(args.stage1_checkpoint_path, "actor.pt"), map_location=device)
    state_dict = ckpt["network_state_dict"]
    if any(k.startswith("_orig_mod.") for k in state_dict):
        state_dict = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
    # Drop env_mlp keys (not present in this actor)
    model_keys = set(actor_net.state_dict().keys())
    state_dict = {k: v for k, v in state_dict.items() if k in model_keys}
    actor_net.load_state_dict(state_dict)
    actor_net.eval()
    print(f"\033[32m[RMA2]\033[0m Loaded Stage 1 actor from {args.stage1_checkpoint_path}")

    # --- Stage 2 adapt_tconv + normalizer ---
    frame_dim = env.obs_size[0] // 3  # 192 // 3 = 64
    adapt_tconv = ProprioAdaptTConv(frame_dim=frame_dim, latent_dim=priv_info_dim).to(device)
    sa_mean_std = RunningMeanStd((30, frame_dim)).to(device)

    adapt_ckpt = torch.load(args.adapt_checkpoint_path, map_location=device)
    adapt_tconv.load_state_dict(adapt_ckpt["adapt_tconv"])
    sa_mean_std.load_state_dict(adapt_ckpt["sa_mean_std"])
    adapt_tconv.eval()
    sa_mean_std.eval()
    print(f"\033[32m[RMA2]\033[0m Loaded adapt_tconv from {args.adapt_checkpoint_path}")

    # --- record loop ---
    frames: list[np.ndarray] = []
    completed_episodes = 0
    episode_returns = np.zeros(args.num_envs)

    while completed_episodes < args.num_episodes:
        frame = env.render()
        if frame is not None:
            if frame.ndim == 4:
                frame = frame[0]
            frames.append(frame.astype(np.uint8))

        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)

        if current_proprio_hist is not None:
            ph = torch.as_tensor(current_proprio_hist, dtype=torch.float32, device=device)
            with torch.no_grad():
                e = torch.tanh(adapt_tconv(sa_mean_std(ph)))
        else:
            e = torch.zeros(obs_t.shape[0], priv_info_dim, device=device)

        augmented_obs = torch.cat([obs_t, e], dim=-1)  # (B, 200)

        with torch.no_grad():
            mean, _ = actor_net.get_mean_and_std(augmented_obs, training=False)
            actions = torch.tanh(mean)

        next_obs, rewards, terminateds, truncateds, infos = env.step(actions.cpu().numpy())
        current_proprio_hist = infos.get("proprio_hist")

        episode_returns += rewards
        dones = np.logical_or(terminateds, truncateds)
        for i in range(args.num_envs):
            if dones[i]:
                completed_episodes += 1
                print(f"Episode {completed_episodes}: return = {episode_returns[i]:.2f}")
                episode_returns[i] = 0.0
                if completed_episodes >= args.num_episodes:
                    break

        obs = next_obs

    env.close()

    if frames:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        imageio.mimwrite(args.output, frames, fps=args.fps)
        print(f"Saved {len(frames)} frames to {args.output}")
    else:
        print("No frames captured.")

    sys.exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record a Stage 2 RMA agent (no priv_info at inference)")
    parser.add_argument("--config_path", type=str, default="./configs")
    parser.add_argument("--config_name", type=str, default="rma2_base")
    parser.add_argument("--overrides", action="append", default=[])
    parser.add_argument("--stage1_checkpoint_path", type=str, required=True, help="Stage 1 checkpoint dir")
    parser.add_argument("--adapt_checkpoint_path", type=str, required=True, help="Stage 2 .pt file")
    parser.add_argument("--output", type=str, default="videos/rma2_recording.mp4")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--num_episodes", type=int, default=3)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--camera_eye", type=float, nargs=3, default=[0.5, 0.5, 0.7], metavar=("X", "Y", "Z"))
    parser.add_argument("--camera_target", type=float, nargs=3, default=[0.0, 0.0, 0.6], metavar=("X", "Y", "Z"))
    args = parser.parse_args()
    record(args)
