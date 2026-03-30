# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint path utilities shared by evaluation helpers."""

from __future__ import annotations

import os
from typing import Callable, Optional


def _resolve_existing_path(path: str) -> Optional[str]:
    """Return absolute path when the input already points to an existing file."""
    expanded = os.path.expanduser(path)
    if os.path.exists(expanded):
        return os.path.abspath(expanded)
    return None


def _candidate_run_dirs(log_root_path: str, load_run: Optional[str]) -> list[str]:
    """Enumerate candidate run directories from CLI load_run input."""
    if not load_run:
        return []

    expanded = os.path.expanduser(load_run)
    candidates: list[str] = []

    if os.path.isabs(expanded):
        candidates.append(expanded)

    if log_root_path:
        candidates.append(os.path.join(log_root_path, expanded))

    # Preserve order while removing duplicates.
    unique: list[str] = []
    seen: set[str] = set()
    for item in candidates:
        norm = os.path.normcase(os.path.abspath(item))
        if norm not in seen:
            seen.add(norm)
            unique.append(item)
    return unique


def resolve_eval_checkpoint_path(
    log_root_path: str,
    load_run: Optional[str],
    load_checkpoint: Optional[str],
    checkpoint_arg: Optional[str],
    retrieve_file_path_func: Optional[Callable[[str], str]] = None,
    get_checkpoint_path_func: Optional[Callable[[str, Optional[str], Optional[str]], str]] = None,
) -> str:
    """Resolve the checkpoint path for eval.py.

    Behavior:
    - Absolute or directly existing checkpoint paths are used as-is.
    - Relative checkpoint names are first resolved against ``load_run`` when provided.
    - If no direct candidate exists, fall back to the original IsaacLab helpers.
    """
    if checkpoint_arg:
        direct_path = _resolve_existing_path(checkpoint_arg)
        if direct_path is not None:
            return direct_path

        for run_dir in _candidate_run_dirs(log_root_path, load_run):
            candidate = _resolve_existing_path(os.path.join(run_dir, checkpoint_arg))
            if candidate is not None:
                return candidate

        if retrieve_file_path_func is not None:
            return retrieve_file_path_func(checkpoint_arg)

        raise FileNotFoundError(f"Unable to resolve checkpoint path: {checkpoint_arg}")

    if get_checkpoint_path_func is not None:
        return get_checkpoint_path_func(log_root_path, load_run, load_checkpoint)

    raise ValueError("get_checkpoint_path_func is required when checkpoint_arg is not provided")


def resolve_training_checkpoint_path(
    log_root_path: str,
    load_run: Optional[str],
    checkpoint_arg: str,
    retrieve_file_path_func: Optional[Callable[[str], str]] = None,
) -> str:
    """Resolve a checkpoint path for policy initialization during training.

    Unlike resume, this helper always requires an explicit checkpoint target.
    It supports:
    - direct absolute or relative-existing checkpoint paths
    - relative checkpoint names resolved against ``load_run``
    - optional custom file retriever fallback
    """

    direct_path = _resolve_existing_path(checkpoint_arg)
    if direct_path is not None:
        return direct_path

    for run_dir in _candidate_run_dirs(log_root_path, load_run):
        candidate = _resolve_existing_path(os.path.join(run_dir, checkpoint_arg))
        if candidate is not None:
            return candidate

    if retrieve_file_path_func is not None:
        return retrieve_file_path_func(checkpoint_arg)

    raise FileNotFoundError(f"Unable to resolve training checkpoint path: {checkpoint_arg}")
