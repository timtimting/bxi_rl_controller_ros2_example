from bxi_example_py_elf3.framework.platform.cpu_affinity import (
    bootstrap_process_scheduling,
    CpuAffinityPlan,
    CpuAffinityRole,
)

_CPU_AFFINITY_PLAN = bootstrap_process_scheduling()

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.utilities import get_rmw_implementation_identifier
from rclpy.qos import QoSProfile, qos_profile_sensor_data
import communication.msg as bxiMsg
import communication.srv as bxiSrv
import nav_msgs.msg
import sensor_msgs.msg
from threading import Event, Lock
import numpy as np

import time
import math
import os
import json
from collections import deque
from pathlib import Path
from std_msgs.msg import String
from geometry_msgs.msg import Pose
from sensor_msgs.msg import JointState
from ament_index_python.packages import get_package_share_directory

from bxi_example_py_elf3.framework.runtime.state_machine import load_state_machine_config
from bxi_example_py_elf3.framework.joints import (
    JointCommandDefaults,
    JointDefault,
    JointLayout,
    JointStateBuffer,
    NamedJointCommandOverride,
)
from bxi_example_py_elf3.framework.mod_api import MotorFrame
from bxi_example_py_elf3.imu_diagnostics import ImuFrameRecorder, format_imu_evidence
from bxi_example_py_elf3.imu_freshness import (
    ImuFallObserver,
    ImuFaultObserver,
    ImuStampObserver,
    assess_imu_frame,
)
from bxi_example_py_elf3.framework.mod_api.geometry import quaternion_to_euler_array
from bxi_example_py_elf3.remote_freshness import RemotePathMonitor
from bxi_example_py_elf3.remote_input import RemoteInputMonitor
from bxi_example_py_elf3.framework.platform import (
    NamedJointStateSource,
    RobotControlRuntime,
    RobotObservation,
)
from bxi_example_py_elf3.framework.platform.communication_trace import (
    CommunicationExecutor,
    CommunicationTrace,
)

robot_name = "elf3"
# Keep these transient guard gains aligned with imu_protection in mod.yaml.
IMU_GUARD_KP = 20.0
IMU_GUARD_KD = 1.0

ELF3_RESET_JOINTS = JointLayout(
    (
        "waist_y_joint",
        "waist_x_joint",
        "waist_z_joint",
        "l_hip_y_joint",
        "l_hip_x_joint",
        "l_hip_z_joint",
        "l_knee_y_joint",
        "l_ankle_y_joint",
        "l_ankle_x_joint",
        "r_hip_y_joint",
        "r_hip_x_joint",
        "r_hip_z_joint",
        "r_knee_y_joint",
        "r_ankle_y_joint",
        "r_ankle_x_joint",
        "l_shoulder_y_joint",
        "l_shoulder_x_joint",
        "l_shoulder_z_joint",
        "l_elbow_y_joint",
        "l_wrist_x_joint",
        "l_wrist_y_joint",
        "l_wrist_z_joint",
        "r_shoulder_y_joint",
        "r_shoulder_x_joint",
        "r_shoulder_z_joint",
        "r_elbow_y_joint",
        "r_wrist_x_joint",
        "r_wrist_y_joint",
        "r_wrist_z_joint",
        "head_z_joint",
        "head_y_joint",
    ),
    label="ELF3 simulation reset",
)

dof_num = ELF3_RESET_JOINTS.dof_num
joint_name = ELF3_RESET_JOINTS.names

# The current ELF3 state message contains two head joints that the original
# 29-joint policies do not command. A future 31-joint policy overrides these
# values naturally because defaults are only applied to omitted joints. Name
# lookup is compiled once; the control-cycle path performs no dictionary lookup.
ELF3_COMMAND_DEFAULTS = JointCommandDefaults(
    {
        "head_y_joint": JointDefault(position=0.0, kp=16.747, kd=1.066),
        "head_z_joint": JointDefault(position=0.0, kp=16.747, kd=1.066),
    }
)


class BxiExample(Node):
    safety_state_names = frozenset({
        "com.bxi.basic_actions/zero_torque",
        "com.bxi.basic_actions/imu_protection",
    })

    def __init__(self, *, cpu_affinity_plan: CpuAffinityPlan):
        super().__init__("bxi_example_py")

        self._shutting_down = Event()
        self._remote_path = RemotePathMonitor(time.monotonic())
        self._remote_path_blocked = False

        # 加载运行参数
        self.load_files()
        self._communication_trace = CommunicationTrace(
            remote_timeout_sec=self._remote_input.timeout_sec
        )
        self._last_remote_suppressed_buttons: tuple[str, ...] = ()
        self._next_remote_sync_warning = 0.0

        self._motor_override = NamedJointCommandOverride(
            timeout_sec=self.motor_override_timeout_sec,
            release_blend_sec=self.motor_override_release_blend_sec,
        )
        self._motor_override_last_names: tuple[str, ...] = ()
        self._motor_override_last_error = ""
        self._motor_override_waiting_for_joints_warned = False

        # 订阅发布ros主题
        self.init_pub_sub()

        # 机器人状态变量(robot states)
        self.omega = np.zeros(3, dtype=np.double)
        self.linear_acceleration = np.zeros(3, dtype=np.double)
        self.quat_xyzw = np.zeros(4, dtype=np.double)
        self.quat_wxyz = np.zeros(4, dtype=np.double)
        self._imu_received = False
        self._imu_last_received_at = None
        self._imu_protection_latched = False
        self._imu_fall_protection_latched = False
        self._imu_latest_frame_fault = False
        self._imu_inference_paused = False
        self._imu_guard_frame: MotorFrame | None = None
        self._imu_guard_zero_torque = False
        self._next_imu_fall_request_warning = 0.0
        self._imu_first_received_logged = False
        self._imu_invalid_frame_count = 0
        self._imu_stamp_observer = ImuStampObserver()
        self._imu_fault_observer = ImuFaultObserver(confirm_count=5)
        self._imu_fall_observer = ImuFallObserver()
        self._imu_frame_recorder = ImuFrameRecorder()
        self._imu_stamp_warning_count = 0
        self._imu_startup_started_at = None
        self._imu_startup_timeout_logged = False
        self._next_imu_startup_warning = 0.0
        self.raw_cmd_vel = np.zeros(3, dtype=np.float32)
        self.pending_remote_events = deque()
        self._next_remote_warning = 0.0
        self._joint_source = NamedJointStateSource(dtype=np.float64)
        self._joint_received = False
        self._bad_joint_state_warned = False
        self._joint_snapshot: JointStateBuffer | None = None
        self._quat_xyzw_snapshot = np.zeros(4, dtype=np.float64)
        self._quat_wxyz_snapshot = np.zeros(4, dtype=np.float64)
        self._omega_snapshot = np.zeros(3, dtype=np.float64)
        self._linear_acceleration_snapshot = np.zeros(3, dtype=np.float64)
        self._cmd_snapshot = np.zeros(3, dtype=np.float32)
        self._observation: RobotObservation | None = None

        # 控制循环初始化
        self.step = 0
        self._second_reset_at = 0.0
        self.runtime = RobotControlRuntime(
            self.state_machine_config,
            built_in_mod_root=self.package_share / "mods",
            command_defaults=ELF3_COMMAND_DEFAULTS,
            ros_node=self,
            platform=self,
            cpu_affinity_plan=cpu_affinity_plan,
            fatal_callback=self._on_control_fatal,
            communication_trace=self._communication_trace,
        )
        self.state_machine_info_timer = None
        if self.state_machine_info_hz > 0.0:
            self.state_machine_info_timer = self.create_timer(
                1.0 / self.state_machine_info_hz,
                self.publish_state_machine_info,
                callback_group=self.status_callback_group,
            )

    def load_files(self):
        self.declare_parameter("/topic_prefix", "default_value")
        self.topic_prefix = (
            self.get_parameter("/topic_prefix").get_parameter_value().string_value
        )

        self.package_share = Path(get_package_share_directory("bxi_example_py_elf3"))
        self.declare_parameter(
            "/state_machine_config",
            os.path.join(
                self.package_share,
                "config",
                "elf3_state_machine.yaml",
            ),
        )
        state_machine_config_path = self.get_parameter("/state_machine_config").value
        self.state_machine_config = load_state_machine_config(state_machine_config_path)

        self.declare_parameter("/state_machine_info_topic", "")
        self.state_machine_info_topic = (
            self.get_parameter("/state_machine_info_topic")
            .get_parameter_value()
            .string_value
        )

        self.declare_parameter("/state_machine_info_hz", 10.0)
        self.state_machine_info_hz = float(
            self.get_parameter("/state_machine_info_hz").value
        )

        self.imu_required = self.topic_prefix.startswith("hardware/")
        self.declare_parameter("/imu_startup_timeout_sec", 15.0)
        self.imu_startup_timeout_sec = float(
            self.get_parameter("/imu_startup_timeout_sec").value
        )
        self.declare_parameter("/imu_runtime_timeout_sec", 0.1)
        self.imu_runtime_timeout_sec = float(
            self.get_parameter("/imu_runtime_timeout_sec").value
        )
        if self.imu_runtime_timeout_sec <= 0.0:
            raise ValueError("/imu_runtime_timeout_sec must be positive")
        self.declare_parameter("/imu_protection_allow_release", False)
        self.imu_protection_allow_release = bool(
            self.get_parameter("/imu_protection_allow_release").value
        )

        self.declare_parameter("/remote_timeout_sec", 0.3)
        self.declare_parameter("/remote_max_age_sec", 0.3)
        self._remote_input = RemoteInputMonitor(
            timeout_sec=float(self.get_parameter("/remote_timeout_sec").value),
            max_age_sec=float(self.get_parameter("/remote_max_age_sec").value),
        )

        self.motor_override_topic = self.topic_prefix + "actuators_cmds_override"
        self.motor_override_timeout_sec = 0.2
        self.motor_override_release_blend_sec = 0.2
        self.motor_override_allow_in_zero_torque = False

    def _on_control_fatal(self, _message: str):
        self._shutting_down.set()
        rclpy.try_shutdown()

    def destroy_node(self):
        self._shutting_down.set()
        runtime = getattr(self, "runtime", None)
        if runtime is not None:
            try:
                runtime.close()
            except Exception as exc:
                self.get_logger().warning(f"control runtime cleanup failed: {exc}")
        return super().destroy_node()

    # ---------------------------------------------------------------------------- #
    #                                    ROS话题部分                                   #
    # ---------------------------------------------------------------------------- #
    def init_pub_sub(self):
        # 订阅和发布主题
        # Independent input streams can be scheduled concurrently; each group
        # still prevents overlapping callbacks that share its local state.
        self.imu_callback_group = MutuallyExclusiveCallbackGroup()
        self.joint_callback_group = MutuallyExclusiveCallbackGroup()
        self.auxiliary_callback_group = MutuallyExclusiveCallbackGroup()
        self.remote_callback_group = MutuallyExclusiveCallbackGroup()
        self.status_callback_group = MutuallyExclusiveCallbackGroup()
        self.remote_diagnostic_callback_group = MutuallyExclusiveCallbackGroup()
        qos = QoSProfile(
            depth=1,
            durability=qos_profile_sensor_data.durability,
            reliability=qos_profile_sensor_data.reliability,
        )

        self.act_pub = self.create_publisher(
            bxiMsg.ActuatorCmds, self.topic_prefix + "actuators_cmds", qos
        )  # CHANGE
        state_machine_info_topic = self.state_machine_info_topic or (
            self.topic_prefix + "state_machine_info"
        )
        self.state_machine_info_pub = self.create_publisher(
            String, state_machine_info_topic, 10
        )

        self.odom_sub = self.create_subscription(
            nav_msgs.msg.Odometry,
            self.topic_prefix + "odom",
            self.odom_callback,
            qos,
            callback_group=self.auxiliary_callback_group,
        )
        self.actuator_sub = self.create_subscription(
            bxiMsg.ActuatorStates,
            self.topic_prefix + "actuator_states",
            self.actuator_callback,
            qos,
            callback_group=self.joint_callback_group,
        )
        self.imu_sub = self.create_subscription(
            sensor_msgs.msg.Imu,
            self.topic_prefix + "imu_data",
            self.imu_callback,
            qos,
            callback_group=self.imu_callback_group,
        )
        self.touch_sub = self.create_subscription(
            bxiMsg.TouchSensor,
            self.topic_prefix + "touch_sensor",
            self.touch_callback,
            qos,
            callback_group=self.auxiliary_callback_group,
        )
        self.joy_sub = self.create_subscription(
            bxiMsg.MotionCommands,
            "motion_commands",
            self.joy_callback,
            qos,
            callback_group=self.remote_callback_group,
        )
        self.remote_freshness_sub = self.create_subscription(
            String, "remote_controller/freshness", self.remote_freshness_callback, 10,
            callback_group=self.remote_diagnostic_callback_group,
        )
        self.remote_freshness_timer = self.create_timer(
            0.1, self.remote_freshness_check, callback_group=self.remote_diagnostic_callback_group,
        )
        self.motor_override_sub = self.create_subscription(
            bxiMsg.ActuatorCmds,
            self.motor_override_topic,
            self.motor_override_callback,
            qos,
            callback_group=self.auxiliary_callback_group,
        )

        self.rest_srv = self.create_client(
            bxiSrv.RobotReset, self.topic_prefix + "robot_reset"
        )
        self.sim_rest_srv = self.create_client(
            bxiSrv.SimulationReset, self.topic_prefix + "sim_reset"
        )

        self.lock_in = Lock()
        self.lock_ou = self.lock_in  # Lock()

        self.get_logger().info(
            "remote input watchdog: "
            f"timeout={self._remote_input.timeout_sec:.3f}s, "
            f"max_source_age={self._remote_input.max_age_ns / 1e9:.3f}s; "
            "timeout clears remote velocity and pending events"
        )
        self.get_logger().info(
            "motor override input: "
            f"topic={self.motor_override_topic}, "
            f"timeout={self.motor_override_timeout_sec:.3f}s, "
            f"release_blend={self.motor_override_release_blend_sec:.3f}s, "
            "allow_in_zero_torque="
            f"{self.motor_override_allow_in_zero_torque}"
        )

    # --------------------------- Runtime 平台适配接口 --------------------------- #

    def startup_step(self, now: float) -> bool:
        """Perform the ELF3-specific two-stage reset before control starts."""
        if self.imu_required and not self._imu_received:
            if self._imu_startup_started_at is None:
                self._imu_startup_started_at = now
            elif (
                not self._imu_startup_timeout_logged
                and now - self._imu_startup_started_at >= self.imu_startup_timeout_sec
            ):
                with self.lock_in:
                    evidence = self._imu_frame_recorder.trigger(
                        "startup_timeout", at=now,
                        timeout_ms=self.imu_startup_timeout_sec * 1000,
                    )
                self.get_logger().error(
                    f"IMU startup timeout: no valid data received on "
                    f"{self.topic_prefix + 'imu_data'} after "
                    f"{self.imu_startup_timeout_sec:.1f}s; "
                    f"evidence={format_imu_evidence(evidence)}"
                )
                self._imu_startup_timeout_logged = True
            if now >= self._next_imu_startup_warning:
                self.get_logger().warning(
                    f"waiting for IMU data on {self.topic_prefix + 'imu_data'} "
                    "before control startup"
                )
                self._next_imu_startup_warning = now + 2.0
            return False
        if self.step == 0:
            if self.robot_reset(1, False):
                self.get_logger().debug("robot reset step 1 requested")
                self._second_reset_at = now + 1.0
                self.step = 1
            return False
        if self.step == 1:
            if now < self._second_reset_at:
                return False
            if not self.robot_reset(2, True):
                return False
            self.get_logger().debug("robot reset step 2 requested")
            self.step = 2
            if self.topic_prefix.find("simulation") != -1:
                self.runtime.request_state(
                    "com.bxi.basic_actions/pd_brake", trigger="AutoPdbreak"
                )
                self.runtime.request_state(
                    "com.bxi.basic_actions/normal", trigger="AutoRelease"
                )
            return False
        with self.lock_in:
            return self._joint_received

    def snapshot_control_inputs(self):
        """Copy the latest ROS inputs into one coherent framework observation."""
        self._check_imu_runtime_safety()
        with self.lock_in:
            now = time.monotonic()
            imu_block_reason = self._imu_block_reason_locked(now)
            inference_blocked = imu_block_reason is not None
            inference_pause_changed = inference_blocked != self._imu_inference_paused
            self._imu_inference_paused = inference_blocked
            if not inference_blocked:
                self._imu_guard_frame = None
                self._imu_guard_zero_torque = False
            remote_expired = self._remote_input.expire(now)
            command_age = self._remote_input.snapshot(now)["receive_age_ms"]
            upstream_unsafe = self._remote_path.input_unsafe(now, command_age)
            if remote_expired or upstream_unsafe:
                self.raw_cmd_vel.fill(0.0)
                self.pending_remote_events.clear()
            if upstream_unsafe:
                self._remote_path_blocked = True
            latest_joints = self._joint_source.view
            if self._joint_snapshot is None:
                self._joint_snapshot = JointStateBuffer(
                    latest_joints.layout,
                    dtype=np.float64,
                )
                self._observation = RobotObservation(
                    joints=self._joint_snapshot.view,
                    quat_xyzw=self._quat_xyzw_snapshot,
                    quat_wxyz=self._quat_wxyz_snapshot,
                    omega=self._omega_snapshot,
                    raw_cmd_vel=self._cmd_snapshot,
                    linear_acceleration=self._linear_acceleration_snapshot,
                )
            self._joint_snapshot.update(
                latest_joints.position,
                latest_joints.velocity,
                timestamp_ns=latest_joints.timestamp_ns,
            )
            np.copyto(self._quat_xyzw_snapshot, self.quat_xyzw)
            self._observation.orientation_unsafe_confirmed = (
                self._imu_fall_observer.snapshot_confirmed()
            )
            self._observation.inference_blocked = inference_blocked
            np.copyto(self._quat_wxyz_snapshot, self.quat_wxyz)
            np.copyto(self._omega_snapshot, self.omega)
            np.copyto(self._linear_acceleration_snapshot, self.linear_acceleration)
            np.copyto(self._cmd_snapshot, self.raw_cmd_vel)
            events = (
                ()
                if inference_blocked
                or self._imu_protection_latched
                or self._imu_fall_protection_latched
                else tuple(self.pending_remote_events)
            )
            self.pending_remote_events.clear()
        if inference_pause_changed:
            if inference_blocked:
                self.get_logger().warning(
                    f"IMU inference paused: reason={imu_block_reason}; "
                    "publishing a guarded motor frame"
                )
            else:
                self.get_logger().info("IMU inference resumed after valid data")
        if remote_expired:
            self.get_logger().warning(
                f"remote input timeout >{self._remote_input.timeout_sec * 1000:.0f}ms: "
                "velocity/buttons cleared; check /motion_commands rate "
                "(repository default: 100Hz full-state)"
            )
        assert self._observation is not None
        return self._observation, events

    def _imu_block_reason_locked(self, now: float) -> str | None:
        if not self.imu_required:
            return None
        if self._imu_fall_protection_latched:
            return "fall_protection"
        if self._imu_protection_latched:
            return "data_protection"
        if self._imu_latest_frame_fault:
            return "invalid_frame"
        if self._imu_fall_observer.consecutive > 0:
            return "orientation_limit"
        if self._imu_last_received_at is None:
            return "missing_frame"
        if now - self._imu_last_received_at >= self.imu_runtime_timeout_sec:
            return "receive_timeout"
        return None

    def _imu_guard_frame_locked(self, joints, state_name: str) -> MotorFrame:
        frame = self._imu_guard_frame
        if frame is None or frame.layout.names != joints.layout.names:
            frame = MotorFrame.empty(joints.layout)
            np.copyto(frame.qpos, joints.position, casting="same_kind")
            self._imu_guard_frame = frame
            self._imu_guard_zero_torque = False
        self._imu_guard_zero_torque |= (
            self._imu_fall_protection_latched
            or state_name == "com.bxi.basic_actions/zero_torque"
        )
        if self._imu_guard_zero_torque:
            frame.kp.fill(0.0)
            frame.kd.fill(0.0)
        else:
            frame.kp.fill(IMU_GUARD_KP)
            frame.kd.fill(IMU_GUARD_KD)
        return frame

    def safety_motor_frame(
        self, observation: RobotObservation, state_name: str
    ) -> MotorFrame:
        with self.lock_in:
            return self._imu_guard_frame_locked(observation.joints, state_name)

    def _check_imu_runtime_safety(self):
        if not self.imu_required:
            return
        now = time.monotonic()
        evidence = None
        timeout_check = None
        with self.lock_in:
            last_received_at = self._imu_last_received_at
            imu_received = self._imu_received
            if (
                imu_received
                and last_received_at is not None
                and not self._imu_protection_latched
                and not self._imu_fall_protection_latched
            ):
                timeout_check = self._imu_fault_observer.check_timeout(
                    now, last_received_at, self.imu_runtime_timeout_sec,
                    self.runtime.scheduler.period_sec,
                )
                if timeout_check is not None and timeout_check[2]:
                    self._imu_protection_latched = True
                    evidence = self._imu_frame_recorder.trigger(
                        "runtime_timeout", at=now,
                        last_valid_age_ms=timeout_check[0] * 1000,
                        timeout_ms=self.imu_runtime_timeout_sec * 1000,
                        confirm_checks=self._imu_fault_observer.confirm_count,
                    )
            protection_latched = self._imu_protection_latched
            fall_latched = self._imu_fall_protection_latched
        if not imu_received or last_received_at is None:
            return
        if timeout_check is not None:
            age_sec, streak, _ = timeout_check
            self.get_logger().warning(
                f"IMU receive timeout: no valid frame for {age_sec:.3f}s; "
                f"check={streak}/{self._imu_fault_observer.confirm_count}"
            )
        if fall_latched:
            target = "com.bxi.basic_actions/zero_torque"
            if self.runtime.current_state_name != target:
                accepted = self.runtime.request_state(
                    target, trigger="imu_fall_confirmed", force=True,
                )
                if not accepted and now >= self._next_imu_fall_request_warning:
                    self._next_imu_fall_request_warning = now + 1.0
                    self.get_logger().error("failed to request locked IMU fall protection")
            return
        if not protection_latched:
            return
        protection_state = "com.bxi.basic_actions/imu_protection"
        current_state = self.runtime.current_state_name
        accepted = True
        if current_state != protection_state:
            accepted = self.runtime.request_state(
                protection_state,
                trigger="imu_timeout",
                force=True,
            )
        if evidence is not None:
            evidence.details["from_state"] = current_state
            evidence.details["request_accepted"] = accepted
            self.get_logger().error(
                f"IMU data timeout: no valid frame for "
                f"{now - last_received_at:.3f}s; "
                f"confirmed across {self._imu_fault_observer.confirm_count} "
                "control checks; requesting locked IMU protection state; "
                f"evidence={format_imu_evidence(evidence)}"
            )
        if not accepted:
            self.get_logger().error(
                "failed to request locked IMU protection state"
            )

    def release_imu_protection(
        self, target_state: str = "com.bxi.basic_actions/zero_torque"
    ) -> bool:
        """Reserved recovery interface; disabled until explicitly enabled."""
        if not self.imu_protection_allow_release:
            self.get_logger().warning(
                "IMU protection release requested but the recovery interface is disabled"
            )
            return False
        if self._imu_fall_protection_latched:
            self.get_logger().warning(
                "IMU protection release rejected because fall protection is latched"
            )
            return False
        with self.lock_in:
            fresh = (
                self._imu_last_received_at is not None
                and time.monotonic() - self._imu_last_received_at
                < self.imu_runtime_timeout_sec
            )
            faulted = (
                self._imu_fault_observer.frame_streak > 0
                or self._imu_fault_observer.timeout_streak > 0
            )
        if not fresh or faulted:
            self.get_logger().warning(
                "IMU protection release rejected because IMU data is stale or abnormal"
            )
            return False
        self._imu_protection_latched = False
        return self.runtime.request_state(
            target_state,
            trigger="imu_recovery",
            force=True,
        )

    def publish_motor_frame(self, frame: MotorFrame, *, safety: bool = False):
        """Convert a framework motor frame into the ELF3 ROS command message."""
        state_name = self.runtime.framework.current_state_name
        with self.lock_in:
            if self._imu_block_reason_locked(time.monotonic()) is not None:
                frame = self._imu_guard_frame_locked(
                    self._joint_source.view, state_name
                )
                safety = True
        if safety:
            self._motor_override.clear()
            self._motor_override_last_names = ()
        override_permitted = not safety and (
            self.motor_override_allow_in_zero_torque
            or state_name != "com.bxi.basic_actions/zero_torque"
        )
        frame = self._motor_override.apply(
            frame,
            now=time.monotonic(),
            permitted=override_permitted,
        )

        msg = bxiMsg.ActuatorCmds()
        msg.header.frame_id = robot_name
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.actuators_name = frame.layout.names
        msg.pos = frame.qpos.tolist()
        msg.vel = frame.vel.tolist()
        msg.torque = frame.torque.tolist()
        msg.kp = frame.kp.tolist()
        msg.kd = frame.kd.tolist()
        self.act_pub.publish(msg)
        self._communication_trace.record(
            "command_tx",
            stamp_ns=msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec,
            ros_now_ns=self.get_clock().now().nanoseconds,
        )

    def communication_summary(self):
        """Small watchdog snapshot without ROS graph queries or control locks."""
        now = time.monotonic()
        return {
            "state": self.runtime.framework.current_state_name,
            "remote_input_fresh": self._remote_input.is_fresh(now),
            "remote_path_status": self._remote_path.status(
                now, self._remote_input.snapshot(now)["receive_age_ms"]
            ),
        }

    def communication_diagnostics(self):
        """Watchdog-only ROS graph queries; never acquire control/input locks."""
        def endpoint_info(endpoints):
            return [{
                "node": e.node_namespace.rstrip("/") + "/" + e.node_name,
                "gid": bytes(e.endpoint_gid).hex(),
                "reliability": e.qos_profile.reliability.name,
                "durability": e.qos_profile.durability.name,
                "history": e.qos_profile.history.name,
                "depth": e.qos_profile.depth,
            } for e in endpoints]

        rx = {}
        for subscription in (
            self.actuator_sub, self.imu_sub, self.joy_sub,
            self.remote_freshness_sub,
        ):
            rx[subscription.topic_name] = endpoint_info(
                self.get_publishers_info_by_topic(subscription.topic_name)
            )
        pressure = {}
        for resource in ("cpu", "io", "memory"):
            try:
                pressure[resource] = Path(f"/proc/pressure/{resource}").read_text().strip()
            except OSError:
                pass
        process = {}
        try:
            for line in Path("/proc/self/status").read_text().splitlines():
                key, _, value = line.partition(":")
                if key in {"VmRSS", "VmHWM", "Threads", "voluntary_ctxt_switches",
                           "nonvoluntary_ctxt_switches"}:
                    process[key] = value.strip()
            process["open_fds"] = len(tuple(Path("/proc/self/fd").iterdir()))
        except OSError:
            pass
        log_filesystem = {}
        try:
            stat = os.statvfs(os.environ.get("BXI_LOG_ROOT", "/var/log/bxi_log"))
            log_filesystem = {
                "free_bytes": stat.f_bavail * stat.f_frsize,
                "free_inodes": stat.f_favail,
            }
        except OSError:
            pass
        now = time.monotonic()
        return {
            "rmw": get_rmw_implementation_identifier(),
            "domain": os.environ.get("ROS_DOMAIN_ID", "0"),
            "localhost_only": os.environ.get("ROS_LOCALHOST_ONLY", "0"),
            "use_sim_time": self.get_parameter("use_sim_time").value,
            "ros_now_ns": self.get_clock().now().nanoseconds,
            "command_topic": self.act_pub.topic_name,
            "command_matched_subscriptions": self.act_pub.get_subscription_count(),
            "command_discovered_subscriptions": endpoint_info(
                self.get_subscriptions_info_by_topic(self.act_pub.topic_name)
            ),
            "input_discovered_publishers": rx,
            "remote_input": self._remote_input.snapshot(time.monotonic()),
            "remote_path": {
                "status": self._remote_path.status(
                    now, self._remote_input.snapshot(now)["receive_age_ms"]
                ),
                "sample": None if self._remote_path.sample is None
                else self._remote_path.sample.copy(),
            },
            "controller_step": self.step,
            "state": self.runtime.framework.current_state_name,
            "load_average": os.getloadavg(),
            "system_pressure": pressure,
            "process": process,
            "log_filesystem": log_filesystem,
        }

    def publish_state_machine_info(self):
        info = self.runtime.snapshot(include_graph=True)
        if info is None:
            return
        with self.lock_in:
            info["remote_input"] = self._remote_input.snapshot(time.monotonic())
        now = self.get_clock().now()
        info.update(
            {
                "stamp": {
                    "sec": int(now.nanoseconds // 1000000000),
                    "nanosec": int(now.nanoseconds % 1000000000),
                },
                "step": int(self.step),
            }
        )

        msg = String()
        msg.data = json.dumps(info, ensure_ascii=False, sort_keys=True)
        self.state_machine_info_pub.publish(msg)

    def robot_reset(self, reset_step, release):
        req = bxiSrv.RobotReset.Request()
        req.reset_step = reset_step
        req.release = release
        req.header.frame_id = robot_name

        while not self.rest_srv.wait_for_service(timeout_sec=0.2):
            if self._shutting_down.is_set() or not rclpy.ok():
                return False
            self.get_logger().info("robot reset service not available; waiting")

        self.rest_srv.call_async(req)
        return True

    def sim_robot_reset(self):
        req = bxiSrv.SimulationReset.Request()
        req.header.frame_id = robot_name

        base_pose = Pose()
        base_pose.position.x = 0.0
        base_pose.position.y = 0.0
        base_pose.position.z = 1.0
        base_pose.orientation.x = 0.0
        base_pose.orientation.y = 0.0
        base_pose.orientation.z = 0.0
        base_pose.orientation.w = 1.0

        joint_state = JointState()
        joint_state.name = joint_name
        joint_state.position = np.zeros(dof_num, dtype=np.float32).tolist()
        joint_state.velocity = np.zeros(dof_num, dtype=np.float32).tolist()
        joint_state.effort = np.zeros(dof_num, dtype=np.float32).tolist()

        req.base_pose = base_pose
        req.joint_state = joint_state

        while not self.sim_rest_srv.wait_for_service(timeout_sec=1.0):
            print("service not available, waiting again...")

        self.sim_rest_srv.call_async(req)

    def joint_callback(self, msg):
        self._update_joint_state(msg.name, msg.position, msg.velocity)

    def actuator_callback(self, msg):
        self._update_joint_state(msg.name, msg.position, msg.velocity)

    def _update_joint_state(self, names, position, velocity):
        error_to_log = None
        first_joint_count = None
        with self.lock_in:
            try:
                latest = self._joint_source.update(
                    names,
                    position,
                    velocity,
                    timestamp_ns=self.get_clock().now().nanoseconds,
                )
            except (TypeError, ValueError) as exc:
                if not self._bad_joint_state_warned:
                    self._bad_joint_state_warned = True
                    error_to_log = str(exc)
            else:
                if not self._joint_received:
                    first_joint_count = latest.layout.dof_num
                self._joint_received = True
                self._bad_joint_state_warned = False
        # Terminal/file output may block. Release the shared input lock before
        # writing logs so other input readers/writers can acquire it.
        if error_to_log is not None:
            self.get_logger().error(f"invalid named joint state: {error_to_log}")
        elif first_joint_count is not None:
            self.get_logger().info(
                "ELF3 state layout initialized from message names: "
                f"{first_joint_count} joints"
            )

    def remote_freshness_callback(self, msg):
        now = time.monotonic()
        with self.lock_in:
            self._remote_path.observe(msg.data, now, self.get_clock().now().nanoseconds)
            age = self._remote_input.snapshot(now)["receive_age_ms"]
            if self._remote_path.input_unsafe(now, age):
                self._remote_path_blocked = True
                self.raw_cmd_vel.fill(0.0)
                self.pending_remote_events.clear()

    def remote_freshness_check(self):
        now = time.monotonic()
        with self.lock_in:
            age = self._remote_input.snapshot(now)["receive_age_ms"]
            report = self._remote_path.report(now, age)
        if report:
            severity, text = report
            if severity == "warning":
                self.get_logger().warning(text)
            else:
                self.get_logger().info(text)

    def joy_callback(self, msg):
        received_at = time.monotonic()
        velocity = (msg.vel_des.x, msg.vel_des.y, msg.yawdot_des)
        stamp_ns = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
        ignored_buttons: tuple[str, ...] = ()
        warn_suppressed = False
        with self.lock_in:
            command_age = self._remote_input.snapshot(received_at)["receive_age_ms"]
            if self._remote_path.input_unsafe(received_at, command_age):
                status = "upstream_" + self._remote_path.status(received_at, command_age)
                self._remote_input.reject(status)
                self._remote_path_blocked = True
                self.raw_cmd_vel.fill(0.0)
                self.pending_remote_events.clear()
            else:
                status = self._remote_input.observe(
                    velocity,
                    stamp_ns=stamp_ns,
                    received_at=received_at,
                    ros_now_ns=self.get_clock().now().nanoseconds,
                )
            accepted = status in ("fresh", "recovered")
            if accepted:
                # Reconnecting with a held button must not replay an old action.
                synchronize = status == "recovered" or self._remote_path_blocked
                if synchronize:
                    self.pending_remote_events.clear()
                events = self.runtime.extract_remote_events(
                    msg, sync_only=self.step < 2 or synchronize
                )
                if synchronize:
                    ignored_buttons = tuple(
                        f"btn_{i}={value}" for i in range(1, 11)
                        if (value := int(getattr(msg, f"btn_{i}", 0))) != 0
                    )
                    warn_suppressed = bool(ignored_buttons) and (
                        ignored_buttons != self._last_remote_suppressed_buttons
                        or received_at >= self._next_remote_sync_warning
                    )
                    if warn_suppressed:
                        self._next_remote_sync_warning = received_at + 5.0
                    self._last_remote_suppressed_buttons = ignored_buttons
                else:
                    self._last_remote_suppressed_buttons = ()
                self.raw_cmd_vel[:] = velocity
                self.pending_remote_events.extend(events)
                self._remote_path_blocked = False
            warn = not accepted and received_at >= self._next_remote_warning
            if warn:
                self._next_remote_warning = received_at + 1.0
        if warn:
            self.get_logger().warning(
                f"dropping remote input: {status}; "
                "check publisher timestamps and clock synchronization"
            )
        if warn_suppressed:
            self.get_logger().warning(
                "remote buttons ignored on input recovery: "
                + ", ".join(ignored_buttons)
                + "; release and press again after motion_commands resumes"
            )

    def motor_override_callback(self, msg):
        """Validate and atomically replace the final named-joint override."""
        names = tuple(msg.actuators_name)
        if not names:
            if msg.pos or msg.kp or msg.kd or msg.vel or msg.torque:
                self._warn_motor_override(
                    "empty actuators_name disables override only when all command "
                    "arrays are also empty"
                )
                return
            self._motor_override.clear()
            if self._motor_override_last_names:
                self.get_logger().info("motor override release requested")
            self._motor_override_last_names = ()
            self._motor_override_last_error = ""
            return

        with self.lock_in:
            waiting_for_joints = not self._joint_received
            imu_blocked = self._imu_block_reason_locked(time.monotonic()) is not None
            warn_waiting = (
                waiting_for_joints
                and not self._motor_override_waiting_for_joints_warned
            )
            if warn_waiting:
                self._motor_override_waiting_for_joints_warned = True
            if not waiting_for_joints:
                robot_layout = self._joint_source.view.layout
        if imu_blocked:
            self._motor_override.clear()
            self._motor_override_last_names = ()
            self._warn_motor_override("IMU data unsafe; override ignored")
            return
        if waiting_for_joints:
            if warn_waiting:
                self.get_logger().warning(
                    "motor override ignored before the robot joint layout "
                    "is initialized"
                )
            return

        try:
            submitted_names = self._motor_override.submit(
                robot_layout,
                names,
                msg.pos,
                msg.kp,
                msg.kd,
                vel=msg.vel if msg.vel else None,
                torque=msg.torque if msg.torque else None,
                received_at=time.monotonic(),
            )
        except (TypeError, ValueError) as exc:
            self._warn_motor_override(str(exc))
            return

        self._motor_override_last_error = ""
        self._motor_override_waiting_for_joints_warned = False
        if submitted_names != self._motor_override_last_names:
            self.get_logger().info(
                "motor override active for joints: "
                f"{submitted_names}"
            )
            self._motor_override_last_names = submitted_names

    def _warn_motor_override(self, message: str):
        if message == self._motor_override_last_error:
            return
        self._motor_override_last_error = message
        self.get_logger().warning(f"invalid motor override: {message}")

    def _log_imu_followups(self, captures):
        for capture in captures:
            self.get_logger().warning(
                f"IMU protection follow-up: {format_imu_evidence(capture)}"
            )

    def imu_callback(self, msg):
        received_at = time.monotonic()
        header = getattr(msg, "header", None)
        stamp = getattr(header, "stamp", None)
        frame_id = getattr(header, "frame_id", "")
        stamp_ns = 0 if stamp is None else stamp.sec * 1_000_000_000 + stamp.nanosec
        stamp_status, stamp_age_ms = "unchecked", None
        if self.imu_required:
            stamp_status, stamp_age_ms = self._imu_stamp_observer.observe(stamp)
        quat = msg.orientation
        avel = msg.angular_velocity
        acceleration = msg.linear_acceleration
        quat_xyzw = (quat.x, quat.y, quat.z, quat.w)
        omega = (avel.x, avel.y, avel.z)
        linear_acceleration = (acceleration.x, acceleration.y, acceleration.z)
        frame_evidence = None
        fall_evidence = None
        fall_confirmed = False
        warn_orientation = False
        with self.lock_in:
            last_valid_at = self._imu_last_received_at
            gap_ms = (
                (received_at - last_valid_at) * 1000
                if last_valid_at is not None else None
            )
            assessment = assess_imu_frame(
                quat_xyzw, omega, linear_acceleration,
                stamp_status=stamp_status, require_stamp=self.imu_required,
                receive_gap_ms=gap_ms,
                timeout_ms=self.imu_runtime_timeout_sec * 1000,
            )
            reasons = assessment.reasons
            accepted = assessment.accepted
            already_protected = (
                self._imu_protection_latched or self._imu_fall_protection_latched
            )
            self._imu_latest_frame_fault = bool(reasons)
            streak, frame_fault_confirmed = self._imu_fault_observer.observe_frame(
                bool(reasons)
            )
            warn_frame = bool(reasons) and not already_protected and (
                streak < self._imu_fault_observer.confirm_count or frame_fault_confirmed
            )
            if not assessment.payload_valid:
                self._imu_invalid_frame_count += 1
            if self.imu_required and stamp_status != "fresh":
                self._imu_stamp_warning_count += 1
            followups = self._imu_frame_recorder.record(
                received_at=received_at, frame_id=frame_id, stamp_ns=stamp_ns,
                stamp_status=stamp_status, transport_age_ms=stamp_age_ms,
                valid=assessment.payload_valid, accepted=accepted,
                reason=",".join(reasons) or None, quat_xyzw=quat_xyzw,
                omega=omega, acceleration=linear_acceleration,
            )
            if accepted:
                self.quat_xyzw[:] = quat_xyzw
                self.quat_wxyz[:] = quat.w, quat.x, quat.y, quat.z
                self.omega[:] = omega
                self.linear_acceleration[:] = linear_acceleration
                self._imu_received = True
                self._imu_last_received_at = received_at
                self._imu_fault_observer.reset_timeout()
                prior_fall_streak = self._imu_fall_observer.consecutive
                fall_confirmed = self._imu_fall_observer.observe(
                    self.quat_xyzw, stamp_ns=stamp_ns
                )
                fall_streak = self._imu_fall_observer.consecutive
                warn_orientation = (
                    fall_streak > 0
                    and prior_fall_streak < self._imu_fall_observer.confirm_frames
                )
            else:
                self._imu_fall_observer.reset_streak()
                fall_streak = 0
            if fall_confirmed:
                self._imu_fall_protection_latched = self.imu_required
                fall_evidence = self._imu_frame_recorder.trigger(
                    "fall_confirmed", at=received_at,
                    threshold_deg=60, confirm_frames=self._imu_fall_observer.confirm_frames,
                )
            if (
                frame_fault_confirmed
                and self.imu_required
                and self._imu_received
                and not self._imu_protection_latched
                and not self._imu_fall_protection_latched
            ):
                self._imu_protection_latched = True
                frame_evidence = self._imu_frame_recorder.trigger(
                    "frame_fault_confirmed", at=received_at,
                    reasons=tuple(reasons), confirm_frames=self._imu_fault_observer.confirm_count,
                    last_valid_age_ms=(
                        (received_at - last_valid_at) * 1000
                        if last_valid_at is not None else None
                    ),
                )
            first_imu_frame = accepted and not self._imu_first_received_logged
            if accepted:
                self._imu_first_received_logged = True
        self._log_imu_followups(followups)
        if warn_frame:
            self.get_logger().warning(
                f"IMU frame anomaly: reason={','.join(reasons)}, "
                f"streak={streak}/{self._imu_fault_observer.confirm_count}, "
                f"frame_id={frame_id}, stamp_ns={stamp_ns}, "
                f"stamp_status={stamp_status}, transport_age_ms={stamp_age_ms}, "
                f"receive_gap_ms={gap_ms}, quat_xyzw={quat_xyzw}, "
                f"quat_norm={assessment.quaternion_norm:.6f}, omega_rad_s={omega}, "
                f"accel_m_s2={linear_acceleration}"
            )
        if frame_evidence is not None:
            self.get_logger().error(
                "IMU frame anomalies confirmed; requesting locked IMU protection state; "
                f"evidence={format_imu_evidence(frame_evidence)}"
            )
        if warn_orientation:
            roll, pitch, _ = quaternion_to_euler_array(quat_xyzw)
            self.get_logger().warning(
                f"IMU orientation over limit: roll={math.degrees(roll):.1f} deg, "
                f"pitch={math.degrees(pitch):.1f} deg, limit=60 deg, "
                f"streak={fall_streak}/{self._imu_fall_observer.confirm_frames}, "
                f"stamp_ns={stamp_ns}, quat_xyzw={quat_xyzw}"
            )
        if fall_confirmed:
            self.get_logger().warning(
                "unsafe orientation confirmed by 5 consecutive IMU frames; "
                f"evidence={format_imu_evidence(fall_evidence)}"
            )
        if first_imu_frame:
            self.get_logger().info(
                f"received first IMU frame on {self.topic_prefix}imu_data"
            )

    def touch_callback(self, _msg):
        pass

    def odom_callback(self, _msg):  # 全局里程计（上帝视角，仅限仿真使用）
        pass


def main(args=None):
    time.sleep(5)

    rclpy.init(args=args)
    node = BxiExample(cpu_affinity_plan=_CPU_AFFINITY_PLAN)

    executor = CommunicationExecutor(
        node._communication_trace, node.get_clock().now, num_threads=3,
        cpu_affinity=_CPU_AFFINITY_PLAN.roles[CpuAffinityRole.COMMUNICATION],
    )
    executor.trace_subscription(node.actuator_sub, "joint_rx")
    executor.trace_subscription(node.imu_sub, "imu_rx")
    executor.trace_subscription(node.joy_sub, "remote_rx")
    try:
        executor.add_node(node)
        node.runtime.attach_executor(executor)
        node.runtime.start()
        executor.spin()
    finally:
        try:
            node.destroy_node()
        finally:
            executor.shutdown()

    # rclpy's signal handler may already have shut down the context before
    # executor.spin() returns. A second shutdown raises RuntimeError.
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == "__main__":
    main()
