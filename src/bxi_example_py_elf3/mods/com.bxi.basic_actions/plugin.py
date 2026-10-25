from bxi_example_py_elf3.framework.mod_api import ModDefinition, ModLoadContext

from .initial_pos_state import InitialPosState
from .pd_brake_state import PdBrakeState
from .zero_torque_state import ZeroTorqueState


def create_mod(context: ModLoadContext) -> ModDefinition:
    return ModDefinition(
        state_factories={
            "zero_torque": lambda state: ZeroTorqueState(
                state.name, state.state_id
            ),
            "pd_brake": lambda state: PdBrakeState(state.name, state.state_id),
            "initial_pos": lambda state: InitialPosState(
                state.name, state.state_id
            ),
        }
    )
