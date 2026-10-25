import importlib
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import yaml

from bxi_example_py_elf3.policies.joints import ELF3_ISAAC_PARAMETERS


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
MOD_ROOT = PACKAGE_ROOT / "mods/com.bxi.basic_actions"
EXPECTED_STATES = {"zero_torque", "pd_brake", "initial_pos"}


def test_only_three_basic_states_are_registered():
    manifest = yaml.safe_load((MOD_ROOT / "mod.yaml").read_text())
    assert set(manifest["states"]) == EXPECTED_STATES
    assert set(manifest["events"]) == EXPECTED_STATES
    assert all(
        route["from"] in EXPECTED_STATES
        and route["to"] in EXPECTED_STATES
        and route["event"] in EXPECTED_STATES
        for route in manifest["routes"]
    )
    assert yaml.safe_load(
        (PACKAGE_ROOT / "config/elf3_state_machine.yaml").read_text()
    )["mod_paths"] == []

    package_name = "_basic_actions_mod_test"
    package = ModuleType(package_name)
    package.__path__ = [str(MOD_ROOT)]
    sys.modules[package_name] = package
    try:
        plugin = importlib.import_module(f"{package_name}.plugin")
        definition = plugin.create_mod(None)
        assert set(definition.state_factories) == EXPECTED_STATES
        pd_state = definition.state_factories["pd_brake"](
            SimpleNamespace(name="pd_brake", state_id=1)
        )
        assert pd_state.required_resources == ()
        frame = pd_state.get_entry_frame(SimpleNamespace())
        assert frame.layout == ELF3_ISAAC_PARAMETERS.layout
        np.testing.assert_array_equal(
            frame.qpos, ELF3_ISAAC_PARAMETERS.default_position
        )
        np.testing.assert_array_equal(frame.kp, ELF3_ISAAC_PARAMETERS.kp)
        np.testing.assert_array_equal(frame.kd, ELF3_ISAAC_PARAMETERS.kd)
    finally:
        for name in list(sys.modules):
            if name == package_name or name.startswith(package_name + "."):
                sys.modules.pop(name, None)
