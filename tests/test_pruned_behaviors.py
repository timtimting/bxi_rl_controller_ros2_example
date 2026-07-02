from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
STATE_CONFIG = ROOT / "src/bxi_example_py_elf3/config/elf3_state_machine.yaml"
REMOTE_CONFIG = ROOT / "src/remote_controller/config/xbox_default.yaml"
DEMO_SOURCE = ROOT / "src/bxi_example_py_elf3/bxi_example_py_elf3/bxi_example_demo.py"
STATE_SOURCE = ROOT / "src/ot_states.py"
MODEL_DIR = ROOT / "src/bxi_example_py_elf3/data"bxi_example_py_elf3/bxi_example_py_elf3/rob


def load_yaml(path: Path):
    with path.open("r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def test_state_machine_exposes_only_requested_behaviors():
    config = load_yaml(STATE_CONFIG)

    assert set(config["states"]) == {
        "normal",
        "zero_torque",
        "pd_brake",
        "initial_pos",
        "dance",
        "back_flip",
        "forward_flip",
        "normal_depth",
    }
    assert set(config["remote_events"]) == {
        "normal_event",
        "zero_torque_event",
        "pd_brake_event",
        "initial_pos_event",
        "dance_event",
        "toggle_dance_pause_event",
        "back_flip_event",
        "forward_flip_event",
        "depth_event",
    }
    assert set(config["speed_profiles"]) == {"normal"}


def test_remote_controller_maps_only_requested_behavior_events():
    config = load_yaml(REMOTE_CONFIG)

    keyboard_signals = config["sources"]["keyboard"]["signals"]
    assert set(keyboard_signals) == {
        "keyboard.vx",
        "keyboard.vy",
        "keyboard.yaw",
        "keyboard.normal",
        "keyboard.dance",
        "keyboard.back_flip",
        "keyboard.forward_flip",
    }

    controls = config["controls"]
    event_controls = {name for name in controls if name.endswith("_event")}
    assert {
        "keyboard.recover_event",
        "keyboard.amp_run_event",
        "keyboard.normal_run_event",
        "keyboard.applause_event",
        "keyboard.hello_event",
    }.isdisjoint(event_controls)

    remote_text = REMOTE_CONFIG.read_text(encoding="utf-8")
    for removed in (
        "recover",
        "amp_run",
        "normal_run",
        "applause",
        "hello",
        "btn_6",
        "btn_7",
        "btn_8",
        "btn_10=3",
        "btn_10=4",
        "btn_10=5",
        "btn_10=6",
    ):
        assert removed not in remote_text


def test_demo_loads_only_requested_policy_assets():
    demo_text = DEMO_SOURCE.read_text(encoding="utf-8")

    for kept in (
        "amp_terrain.onnx",
        "shuishou.npz",
        "shuishou.onnx",
        "back_flip.npz",
        "back_flip.onnx",
        "forward_flip.npz",
        "forward_flip.onnx",
        "normal_depth.onnx",
    ):
        assert kept in demo_text

    for removed in (
        "recover.npz",
        "recover.onnx",
        "amp_run.onnx",
        "model_normal.onnx",
        "ballet.npz",
        "ballet.onnx",
        "withoutarm.onnx",
    ):
        assert removed not in demo_text


def test_removed_state_classes_are_not_registered():
    state_text = STATE_SOURCE.read_text(encoding="utf-8")

    for removed in (
        "RecoverState",
        "AmpRunState",
        "NormalRunState",
        "BalletState",
        "ApplauseState",
        "HelloState",
        "HandPlayBackState",
    ):
        assert f"class {removed}" not in state_text


def test_only_requested_policy_assets_remain_tracked():
    kept_assets = {
        "isaaclab_model/amp_terrain.onnx",
        "isaaclab_model/shuishou.npz",
        "isaaclab_model/shuishou.onnx",
        "isaaclab_model/back_flip.npz",
        "isaaclab_model/back_flip.onnx",
        "isaaclab_model/forward_flip.npz",
        "isaaclab_model/forward_flip.onnx",
        "isaaclab_model/normal_depth.onnx",
    }
    removed_assets = {
        "isaaclab_model/amp_run.onnx",
        "isaaclab_model/applause.pkl",
        "isaaclab_model/ballet.npz",
        "isaaclab_model/ballet.onnx",
        "isaaclab_model/withoutarm.onnx",
        "mjlab_model/model_normal.onnx",
        "mjlab_model/recover.npz",
        "mjlab_model/recover.onnx",
    }

    for asset in kept_assets:
        assert (MODEL_DIR / asset).is_file()
    for asset in removed_assets:
        assert not (MODEL_DIR / asset).exists()
