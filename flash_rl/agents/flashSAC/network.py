import math
from typing import Optional

import torch
import torch.nn as nn

from flash_rl.agents.flashSAC.layer import (
    EnsembleCategoricalValue,
    EnsembleFlashSACBlock,
    EnsembleFlashSACEmbedder,
    EnsembleUnitRMSNorm,
    FlashSACBlock,
    FlashSACEmbedder,
    NormalTanhPolicy,
    UnitRMSNorm,
)


class RunningMeanStd(nn.Module):
    """Welford online running mean/variance normalizer.

    Updates running stats only in training mode (.train()); in eval mode (.eval()) only normalizes.
    """

    def __init__(self, shape: tuple[int, ...], eps: float = 1e-5):
        super().__init__()
        self.register_buffer("mean", torch.zeros(shape, dtype=torch.float32))
        self.register_buffer("var", torch.ones(shape, dtype=torch.float32))
        self.register_buffer("count", torch.tensor(0.0, dtype=torch.float64))
        self.eps = eps

    def update(self, x: torch.Tensor) -> None:
        batch = x.reshape(-1, *self.mean.shape)
        n = batch.shape[0]
        batch_mean = batch.mean(0)
        batch_var = batch.var(0, unbiased=False)
        total = self.count + n
        delta = batch_mean - self.mean
        self.mean = self.mean + delta * (n / total)
        self.var = (self.var * self.count + batch_var * n + delta.pow(2) * self.count * n / total) / total
        self.count = total

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / (self.var.sqrt() + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            self.update(x.detach())
        return self.normalize(x)


class ProprioAdaptTConv(nn.Module):
    """Temporal convolution network: proprio_hist (B, T, frame_dim) → latent (B, latent_dim).

    Architecture from RMA (Qi et al. 2022): per-frame channel transform, then causal temporal
    aggregation with three Conv1d layers, followed by a linear projection.
    T=30 frames → after Conv1d(k=9,s=2), Conv1d(k=5), Conv1d(k=5): 3 output frames.
    """

    def __init__(self, frame_dim: int, latent_dim: int = 8):
        super().__init__()
        self.channel_transform = nn.Sequential(
            nn.Linear(frame_dim, frame_dim),
            nn.ReLU(inplace=True),
            nn.Linear(frame_dim, frame_dim),
            nn.ReLU(inplace=True),
        )
        self.temporal_aggregation = nn.Sequential(
            nn.Conv1d(frame_dim, frame_dim, kernel_size=9, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(frame_dim, frame_dim, kernel_size=5, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(frame_dim, frame_dim, kernel_size=5, stride=1),
            nn.ReLU(inplace=True),
        )
        self.low_dim_proj = nn.Linear(frame_dim * 3, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_transform(x)       # (B, T, frame_dim)
        x = x.permute(0, 2, 1)              # (B, frame_dim, T)
        x = self.temporal_aggregation(x)    # (B, frame_dim, 3)
        return self.low_dim_proj(x.flatten(1))  # (B, latent_dim)


def build_env_mlp(input_dim: int, units: list[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    in_dim = input_dim
    for out_dim in units[:-1]:
        layers += [nn.Linear(in_dim, out_dim), nn.ELU()]
        in_dim = out_dim
    layers += [nn.Linear(in_dim, units[-1]), nn.Tanh()]
    return nn.Sequential(*layers)


class FlashSACActor(nn.Module):
    def __init__(
        self,
        num_blocks: int,
        input_dim: int,
        hidden_dim: int,
        action_dim: int,
        priv_info_dim: int = 0,
        env_mlp: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.priv_info_dim = priv_info_dim
        self.env_mlp = env_mlp
        self.embedder = FlashSACEmbedder(input_dim=input_dim, hidden_dim=hidden_dim)
        self.encoder = nn.ModuleList([FlashSACBlock(hidden_dim) for _ in range(num_blocks)])
        self.post_norm = UnitRMSNorm(hidden_dim)
        self.predictor = NormalTanhPolicy(hidden_dim=hidden_dim, action_dim=action_dim)

    def _encode_priv(self, observations: torch.Tensor) -> torch.Tensor:
        if self.env_mlp is not None and self.priv_info_dim > 0:
            policy_obs = observations[..., : -self.priv_info_dim]
            priv_info = observations[..., -self.priv_info_dim :]
            e = self.env_mlp(priv_info)
            return torch.cat([policy_obs, e], dim=-1)
        return observations

    def get_mean_and_std(
        self,
        observations: torch.Tensor,
        training: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self._encode_priv(observations)
        x = self.embedder(x, training)
        for block in self.encoder:
            x = block(x, training)
        x = self.post_norm(x)
        mean, std = self.predictor.get_mean_and_std(x, training)
        return mean, std

    def forward(
        self,
        observations: torch.Tensor,
        training: bool,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        x = self._encode_priv(observations)
        x = self.embedder(x, training)
        for block in self.encoder:
            x = block(x, training)
        x = self.post_norm(x)
        actions, info = self.predictor(x, training)
        return actions, info


class FlashSACDoubleCritic(nn.Module):
    """
    Double-Q for Clipped Double Q-learning.
    https://arxiv.org/pdf/1802.09477v3

    Fuses N parallel critic networks into single batched operations.
    All internal computation uses (N, batch, dim) tensor layout.
    """

    def __init__(
        self,
        num_blocks: int,
        input_dim: int,
        hidden_dim: int,
        num_bins: int,
        min_v: float,
        max_v: float,
        num_qs: int = 2,
        priv_info_dim: int = 0,
        env_mlp: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.num_qs = num_qs
        self.priv_info_dim = priv_info_dim
        self.env_mlp = env_mlp

        self.embedder = EnsembleFlashSACEmbedder(num_qs, input_dim, hidden_dim)
        self.encoder = nn.ModuleList([EnsembleFlashSACBlock(num_qs, hidden_dim) for _ in range(num_blocks)])
        self.post_norm = EnsembleUnitRMSNorm(num_qs, hidden_dim)
        self.predictor = EnsembleCategoricalValue(
            num_ensemble=num_qs,
            hidden_dim=hidden_dim,
            num_bins=num_bins,
            min_v=min_v,
            max_v=max_v,
        )

    def _encode_priv(self, observations: torch.Tensor) -> torch.Tensor:
        if self.env_mlp is not None and self.priv_info_dim > 0:
            policy_obs = observations[..., : -self.priv_info_dim]
            priv_info = observations[..., -self.priv_info_dim :]
            e = self.env_mlp(priv_info)
            return torch.cat([policy_obs, e], dim=-1)
        return observations

    def forward(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        training: bool,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        obs = self._encode_priv(observations)
        x = torch.cat((obs, actions), dim=-1)  # [B, in_dim]
        x = x.unsqueeze(0).expand(self.num_qs, -1, -1)  # [num_qs, B, in_dim]
        x = self.embedder(x, training)
        for block in self.encoder:
            x = block(x, training)
        x = self.post_norm(x)
        qs, infos = self.predictor(x, training)
        return qs, infos


class FlashSACTemperature(nn.Module):
    def __init__(self, initial_value: float = 0.01):
        super().__init__()
        self.log_temp = nn.Parameter(torch.tensor([math.log(initial_value)], dtype=torch.float32))

    def forward(self) -> torch.Tensor:
        return torch.exp(self.log_temp)
