from __future__ import annotations

from typing import TYPE_CHECKING

from bxi_example_py_elf3.framework.mod_api import RobotControlState
from bxi_example_py_elf3.policies.joints import ELF3_ISAAC_PARAMETERS
from bxi_example_py_elf3.framework.mod_api.transition import (
    EntryFrameProvider,
    MotorFrame,
    RunningFrameProvider,
)

if TYPE_CHECKING:
    from bxi_example_py_elf3.framework.mod_api import RobotControlContext


class PdBrakeState(RobotControlState, EntryFrameProvider, RunningFrameProvider):
    def _frame(self, ctx: RobotControlContext) -> MotorFrame:
        params = ELF3_ISAAC_PARAMETERS
        return self._motor_frame(
            ctx,
            params.default_position,
            params.kp,
            params.kd,
            layout=params.layout,
        )

    def get_entry_frame(self, ctx: RobotControlContext) -> MotorFrame:
        return self._frame(ctx)

    def sample_running_frame(
        self, ctx: RobotControlContext, dt: float, *, advance: bool
    ) -> MotorFrame:
        return self._frame(ctx)

    def on_update(self, ctx: RobotControlContext, dt: float) -> None:
        self._apply_frame(ctx, self._frame(ctx))
