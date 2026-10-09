from pathlib import Path
from types import SimpleNamespace
from xml.etree import ElementTree

import numpy as np
import pytest

from bxi_example_py_elf3.framework.mod_api.hardware_launch import (
    DEFAULT_EXTRA_MOTOR_DISABLE_MASK,
)
from bxi_example_py_elf3.policies.joints import ELF3_POLICY_JOINTS


XML_PATH = (
    Path(__file__).resolve().parents[1]
    / "data/mujoco_simulation/elf3_half.xml"
)
EXPECTED_JOINTS = tuple(
    name
    for index, name in enumerate(ELF3_POLICY_JOINTS.names)
    if not DEFAULT_EXTRA_MOTOR_DISABLE_MASK & (1 << index)
) + ("head_z_joint", "head_y_joint")


def test_half_model_has_only_connected_leg_joints_and_keeps_head():
    root = ElementTree.parse(XML_PATH).getroot()
    assert root.get("model") == "elf3"
    joints = tuple(
        joint.get("name")
        for joint in root.findall(".//worldbody//joint")
        if joint.get("name") != "world_joint"
    )
    motors = tuple(motor.get("name") for motor in root.findall(".//actuator/motor"))
    assert joints == motors == EXPECTED_JOINTS
    assert len(joints) == 21

    for side in ("l", "r"):
        hip = root.find(f".//body[@name='{side}_hip_y_link']")
        assert hip is not None
        assert hip.find("inertial") is not None
        assert hip.find("joint") is not None
        assert hip.find(".//geom") is None
        for link in ("hip_x", "hip_z", "knee_y", "ankle_y", "ankle_x"):
            assert root.find(f".//body[@name='{side}_{link}_link']") is None
        for link in ("hip_y", "hip_x", "hip_z", "knee_y", "ankle_y", "ankle_x"):
            assert root.find(f".//asset/mesh[@name='{side}_{link}_link']") is None


def test_half_model_compiles_with_mujoco():
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_path(str(XML_PATH.resolve()))
    assert model.nu == len(EXPECTED_JOINTS)
    assert model.njnt == len(EXPECTED_JOINTS) + 1  # Base free joint.
    assert tuple(
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, index)
        for index in range(model.nu)
    ) == EXPECTED_JOINTS


def test_head_motor_commands_are_simulation_only(monkeypatch):
    pytest.importorskip("rclpy")
    pytest.importorskip("communication.msg")
    from builtin_interfaces.msg import Time

    from bxi_example_py_elf3.framework.joints import JointLayout
    from bxi_example_py_elf3.framework.platform import cpu_affinity

    monkeypatch.setattr(cpu_affinity, "bootstrap_process_scheduling", lambda: None)
    from bxi_example_py_elf3 import bxi_example_demo

    assert ElementTree.parse(XML_PATH).getroot().get("model") == bxi_example_demo.robot_name

    names = ELF3_POLICY_JOINTS.names + ("head_z_joint", "head_y_joint")
    positions = np.zeros(len(names), dtype=np.float32)
    frame = SimpleNamespace(
        layout=JointLayout.create(names),
        qpos=positions,
        vel=positions,
        torque=positions,
        kp=positions,
        kd=positions,
    )
    messages = []
    platform = SimpleNamespace(
        topic_prefix="simulation/",
        runtime=SimpleNamespace(
            framework=SimpleNamespace(
                current_state_name="com.bxi.basic_actions/pd_brake"
            )
        ),
        motor_override_allow_in_zero_torque=False,
        _motor_override=SimpleNamespace(apply=lambda frame, **_: frame),
        get_clock=lambda: SimpleNamespace(
            now=lambda: SimpleNamespace(to_msg=lambda: Time())
        ),
        act_pub=SimpleNamespace(publish=messages.append),
    )

    bxi_example_demo.BxiExample.publish_motor_frame(platform, frame)
    assert tuple(messages[-1].actuators_name) == EXPECTED_JOINTS

    platform.topic_prefix = "hardware/"
    bxi_example_demo.BxiExample.publish_motor_frame(platform, frame)
    assert tuple(messages[-1].actuators_name) == EXPECTED_JOINTS[:-2]
