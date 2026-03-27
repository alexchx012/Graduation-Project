#!/usr/bin/env python3
"""Scenario definitions for Phase 4 MORL evaluation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScenarioSpec:
    """Frozen stage-4 scenario definition."""

    scenario_id: str
    scenario_name: str
    terrain_mode: str
    command_vx: float
    disturbance_mode: str
    analysis_group: str


SCENARIO_SPECS: dict[str, ScenarioSpec] = {
    "S1": ScenarioSpec(
        scenario_id="S1",
        scenario_name="flat_mid_speed",
        terrain_mode="plane",
        command_vx=1.0,
        disturbance_mode="none",
        analysis_group="main",
    ),
    "S2": ScenarioSpec(
        scenario_id="S2",
        scenario_name="flat_high_speed",
        terrain_mode="plane",
        command_vx=1.5,
        disturbance_mode="none",
        analysis_group="stress",
    ),
    "S3": ScenarioSpec(
        scenario_id="S3",
        scenario_name="uphill_20deg",
        terrain_mode="slope_up",
        command_vx=0.8,
        disturbance_mode="none",
        analysis_group="main",
    ),
    "S4": ScenarioSpec(
        scenario_id="S4",
        scenario_name="downhill_20deg",
        terrain_mode="slope_down",
        command_vx=0.6,
        disturbance_mode="none",
        analysis_group="main",
    ),
    "S5": ScenarioSpec(
        scenario_id="S5",
        scenario_name="stairs_15cm",
        terrain_mode="stairs_15cm",
        command_vx=0.5,
        disturbance_mode="none",
        analysis_group="main",
    ),
    "S6": ScenarioSpec(
        scenario_id="S6",
        scenario_name="lateral_disturbance",
        terrain_mode="plane",
        command_vx=0.8,
        disturbance_mode="velocity_push_equivalent",
        analysis_group="stress",
    ),
}


def list_scenarios() -> list[str]:
    """Return the supported stage-4 scenario ids in a stable order."""

    return list(SCENARIO_SPECS.keys())


def get_scenario_spec(scenario_id: str) -> ScenarioSpec:
    """Return a frozen scenario definition by id."""

    normalized = scenario_id.strip().upper()
    if normalized not in SCENARIO_SPECS:
        raise KeyError(f"Unknown scenario id: {scenario_id}")
    return SCENARIO_SPECS[normalized]
