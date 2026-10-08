from bxi_example_py_elf3.framework.mod_api import (
    ModDefinition,
    ModLoadContext,
    ResourceKey,
    ResourceLoadContext,
)

from .applause_state import ApplauseState, PlaybackClip, load_clip
from .imu_protection_state import ImuProtectionState
from .initial_pos_state import InitialPosState
from .pd_brake_state import PdBrakeState
from .zero_torque_state import ZeroTorqueState


APPLAUSE_CLIP = ResourceKey[PlaybackClip]("com.bxi.basic_actions/applause_clip")


def _load_applause_clip(context: ResourceLoadContext) -> PlaybackClip:
    return load_clip(
        context.asset("assets/applause.pkl"),
        start_frame=600,
        tail_trim_frames=600,
    )


def create_mod(context: ModLoadContext) -> ModDefinition:
    context.register_resource(APPLAUSE_CLIP, _load_applause_clip, policy="on_demand")
    applause_clip = context.resource(APPLAUSE_CLIP)

    return ModDefinition(
        state_factories={
            "zero_torque": lambda state: ZeroTorqueState(
                state.name, state.state_id
            ),
            "imu_protection": lambda state: ImuProtectionState(
                state.name,
                state.state_id,
                kp=state.float_param("kp", 20.0),
                kd=state.float_param("kd", 1.0),
            ),
            "pd_brake": lambda state: PdBrakeState(state.name, state.state_id),
            "initial_pos": lambda state: InitialPosState(
                state.name, state.state_id
            ),
            "applause": lambda state: ApplauseState(
                state.name, state.state_id, applause_clip
            ),
        }
    )
