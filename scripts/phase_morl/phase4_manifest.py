# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Helpers for Phase 4 manifest-driven evaluation and analysis."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
MANIFEST_DIR = SCRIPT_DIR / "manifests"
DEFAULT_PHASE4_MAIN_MANIFEST = MANIFEST_DIR / "phase4_main_manifest.json"
DEFAULT_PHASE4_ANALYSIS_CONFIG = MANIFEST_DIR / "phase4_analysis_config.json"

_REQUIRED_ENTRY_FIELDS = (
    "family",
    "policy_id",
    "canonical_seed",
    "run_dir",
    "checkpoint",
    "task",
    "output_stem",
    "evidence_layer",
    "official_hv_eligible",
)


@dataclass(frozen=True)
class Phase4ManifestEntry:
    family: str
    policy_id: str
    canonical_seed: int
    run_dir: str
    checkpoint: str
    task: str
    output_stem: str
    evidence_layer: str
    official_hv_eligible: bool
    source_state: str = "active"


def _read_manifest_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Phase 4 manifest not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_phase4_manifest(path: Path | str = DEFAULT_PHASE4_MAIN_MANIFEST) -> list[Phase4ManifestEntry]:
    manifest_path = Path(path)
    data = _read_manifest_json(manifest_path)
    entries_raw = data.get("entries")
    if not isinstance(entries_raw, list):
        raise ValueError(f"Invalid phase4 manifest format in {manifest_path}: missing 'entries' list")

    entries: list[Phase4ManifestEntry] = []
    for idx, raw in enumerate(entries_raw):
        if not isinstance(raw, dict):
            raise ValueError(f"Invalid manifest entry at index {idx}: expected object")

        missing = [field for field in _REQUIRED_ENTRY_FIELDS if field not in raw]
        if missing:
            raise ValueError(f"Manifest entry {idx} missing required fields: {missing}")

        entries.append(
            Phase4ManifestEntry(
                family=str(raw["family"]),
                policy_id=str(raw["policy_id"]),
                canonical_seed=int(raw["canonical_seed"]),
                run_dir=str(Path(raw["run_dir"])),
                checkpoint=str(raw["checkpoint"]),
                task=str(raw["task"]),
                output_stem=str(raw["output_stem"]),
                evidence_layer=str(raw["evidence_layer"]),
                official_hv_eligible=bool(raw["official_hv_eligible"]),
                source_state=str(raw.get("source_state", "active")),
            )
        )

    return entries


def filter_phase4_manifest_entries(
    entries: list[Phase4ManifestEntry],
    *,
    policy_ids: set[str] | None = None,
    seeds: set[int] | None = None,
    families: set[str] | None = None,
) -> list[Phase4ManifestEntry]:
    filtered = []
    for entry in entries:
        if policy_ids and entry.policy_id.upper() not in policy_ids:
            continue
        if seeds and entry.canonical_seed not in seeds:
            continue
        if families and entry.family.lower() not in families:
            continue
        filtered.append(entry)
    return filtered


def load_phase4_analysis_config(path: Path | str = DEFAULT_PHASE4_ANALYSIS_CONFIG) -> dict:
    return _read_manifest_json(Path(path))

