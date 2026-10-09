from __future__ import annotations

import numpy as np

from bxi_example_py_elf3.framework.mod_api import (
    ModDefinition,
    ModLoadContext,
    ResourceKey,
    ResourceLoadContext,
)

from .state import ArmRangeState, RunningState, RunningSession, load_trajectory


TRAJECTORY = ResourceKey[np.ndarray]("com.bxi.half_demo/trajectory")


def _load_trajectory(context: ResourceLoadContext) -> np.ndarray:
    return load_trajectory(context.asset("assets/data.txt"))


def create_mod(context: ModLoadContext) -> ModDefinition:
    context.register_resource(TRAJECTORY, _load_trajectory, policy="on_demand")
    trajectory = context.resource(TRAJECTORY)
    session = RunningSession()
    return ModDefinition(
        state_factories={
            "running": lambda state: RunningState(
                state.name, state.state_id, session, trajectory
            ),
            "arm_range": lambda state: ArmRangeState(state.name, state.state_id, session),
        }
    )
