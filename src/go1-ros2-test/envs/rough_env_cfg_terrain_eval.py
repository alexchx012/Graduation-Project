# Canonical source: src/go1-ros2-test/envs/rough_env_cfg_terrain_eval.py
# Deployed to: robot_lab/.../config/quadruped/unitree_go1_ros2/rough_env_cfg_terrain_eval.py (runtime)
# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Phase 3 terrain-specific evaluation configs (PLAY only).

Each config inherits UnitreeGo1Ros2CmdRoughEnvCfg_PLAY and replaces all
sub_terrains with a single target terrain at a fixed difficulty level.
The model checkpoint, observation space, and network architecture remain
identical to the Rough baseline — only the terrain geometry changes.

Configs
-------
Slope10   hf_pyramid_slope   slope_range=(0.176, 0.176)   tan(10°)
Slope20   hf_pyramid_slope   slope_range=(0.364, 0.364)   tan(20°)
Stairs10  pyramid_stairs     step_height_range=(0.10, 0.10)
Stairs15  pyramid_stairs     step_height_range=(0.15, 0.15)  OOD
"""

from __future__ import annotations

import isaaclab.terrains as terrain_gen
from isaaclab.utils import configclass

from .rough_env_cfg import UnitreeGo1Ros2CmdRoughEnvCfg_PLAY


# ---------------------------------------------------------------------------
# Slope 10°
# ---------------------------------------------------------------------------

@configclass
class UnitreeGo1Ros2CmdRoughEnvCfg_Slope10_PLAY(UnitreeGo1Ros2CmdRoughEnvCfg_PLAY):
    """Terrain-specific eval: 10° slope only (in-distribution)."""

    def __post_init__(self):
        super().__post_init__()
        tg = self.scene.terrain.terrain_generator
        tg.sub_terrains = {
            "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
                proportion=1.0,
                slope_range=(0.176, 0.176),  # tan(10°) ≈ 0.176
                platform_width=2.0,
                border_width=0.25,
            ),
        }
        tg.curriculum = False


# ---------------------------------------------------------------------------
# Slope 20°
# ---------------------------------------------------------------------------

@configclass
class UnitreeGo1Ros2CmdRoughEnvCfg_Slope20_PLAY(UnitreeGo1Ros2CmdRoughEnvCfg_PLAY):
    """Terrain-specific eval: 20° slope only (in-distribution)."""

    def __post_init__(self):
        super().__post_init__()
        tg = self.scene.terrain.terrain_generator
        tg.sub_terrains = {
            "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
                proportion=1.0,
                slope_range=(0.364, 0.364),  # tan(20°) ≈ 0.364
                platform_width=2.0,
                border_width=0.25,
            ),
        }
        tg.curriculum = False


# ---------------------------------------------------------------------------
# Stairs 10cm
# ---------------------------------------------------------------------------

@configclass
class UnitreeGo1Ros2CmdRoughEnvCfg_Stairs10_PLAY(UnitreeGo1Ros2CmdRoughEnvCfg_PLAY):
    """Terrain-specific eval: 10cm stairs only (at training upper bound)."""

    def __post_init__(self):
        super().__post_init__()
        tg = self.scene.terrain.terrain_generator
        tg.sub_terrains = {
            "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
                proportion=1.0,
                step_height_range=(0.10, 0.10),
                step_width=0.3,
                platform_width=3.0,
                border_width=1.0,
                holes=False,
            ),
        }
        tg.curriculum = False


# ---------------------------------------------------------------------------
# Stairs 15cm (OOD — exceeds training grid_height_range upper bound 0.1m)
# ---------------------------------------------------------------------------

@configclass
class UnitreeGo1Ros2CmdRoughEnvCfg_Stairs15_PLAY(UnitreeGo1Ros2CmdRoughEnvCfg_PLAY):
    """Terrain-specific eval: 15cm stairs only (out-of-distribution).

    The Go1 baseline trains with grid_height_range=(0.025, 0.1).
    15cm exceeds this upper bound; degraded performance is expected.
    """

    def __post_init__(self):
        super().__post_init__()
        tg = self.scene.terrain.terrain_generator
        tg.sub_terrains = {
            "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
                proportion=1.0,
                step_height_range=(0.15, 0.15),
                step_width=0.3,
                platform_width=3.0,
                border_width=1.0,
                holes=False,
            ),
        }
        tg.curriculum = False

