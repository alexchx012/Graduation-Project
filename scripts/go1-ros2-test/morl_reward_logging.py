# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Helpers for logging MORL reward raw values and weighted contributions."""

from __future__ import annotations

from collections.abc import Sequence


def build_reward_contribution_log(
    reward_manager,
    max_episode_length_s: float,
    env_ids: Sequence[int] | slice | None = None,
    term_names: Sequence[str] | None = None,
) -> dict[str, float]:
    """Build contribution diagnostics from RewardManager episodic sums."""

    import torch

    if env_ids is None:
        env_ids = slice(None)

    selected_terms = term_names or tuple(reward_manager._episode_sums.keys())
    extras: dict[str, float] = {}
    for term_name in selected_terms:
        episode_sums = reward_manager._episode_sums.get(term_name)
        if episode_sums is None:
            continue
        term_cfg = reward_manager.get_term_cfg(term_name)
        weight = float(term_cfg.weight)
        if weight == 0.0:
            continue

        weighted_avg = torch.mean(episode_sums[env_ids]) / max_episode_length_s
        raw_avg = weighted_avg / weight
        extras[f"Episode_RewardWeighted/{term_name}"] = float(weighted_avg.cpu().item())
        extras[f"Episode_RewardRaw/{term_name}"] = float(raw_avg.cpu().item())
    return extras


def attach_reward_contribution_logging(
    reward_manager,
    max_episode_length_s: float,
    term_names: Sequence[str] | None = None,
):
    """Patch RewardManager.reset to emit raw + weighted reward diagnostics."""

    original_reset = reward_manager.reset

    def _wrapped_reset(env_ids=None):
        extras = build_reward_contribution_log(
            reward_manager=reward_manager,
            max_episode_length_s=max_episode_length_s,
            env_ids=env_ids,
            term_names=term_names,
        )
        base_extras = original_reset(env_ids)
        base_extras.update(extras)
        return base_extras

    reward_manager.reset = _wrapped_reset
    return reward_manager
