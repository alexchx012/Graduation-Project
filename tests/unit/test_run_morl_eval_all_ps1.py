# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Static checks for the MORL PowerShell batch evaluation script."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts" / "phase_morl" / "run_morl_eval_all.ps1"


def test_morl_eval_all_ps1_exists_and_has_required_features():
    assert SCRIPT_PATH.exists(), f"Missing script: {SCRIPT_PATH}"

    content = SCRIPT_PATH.read_text(encoding="utf-8")

    required_snippets = [
        "conda activate env_isaaclab",
        "Start-Process -FilePath \"wsl\"",
        "go1_cmd_script_node.py",
        "run_morl_eval.py",
        "summary_json",
        "try {",
        "finally {",
        "$ErrorActionPreference = \"Stop\"",
        "Get-ChildItem",
        "pkill -f go1_cmd_script_node.py",
        "if ($LASTEXITCODE -ne 0)",
        "[switch]$DryRun",
        "[DRY-RUN]",
    ]

    for snippet in required_snippets:
        assert snippet in content, f"Expected snippet not found: {snippet}"
