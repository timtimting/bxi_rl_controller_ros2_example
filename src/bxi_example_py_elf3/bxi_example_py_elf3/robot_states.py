import asyncio
import math
import os
import pickle
import queue
import threading
import time
import weakref
from typing import TYPE_CHECKING, Any, Callable, Optional

import numpy as np

from ament_index_python.packages import get_package_share_path
from bxi_example_py_elf3.utils.robot_state_base import MotorFrame, RobotControlState
from bxi_example_py_elf3.utils.state_machine import StateBehavior, TransitionProfile

if TYPE_CHECKING:
    from bxi_example_py_elf3.bxi_example_demo import BxiExample
else:
    BxiExample = Any


class _BleGattWriter:
    def __init__(
        self,
        address: str,
        characteristic: str,
        payload: bytes,
        write_with_response: bool,
        logger: Optional[Callable[[str, str], None]] = None,
        connect_timeout: float = 10.0,
        reconnect_delay: float = 1.0,
    ):
        self.address = address
        self.characteristic = characteristic
        self.payload = payload
        self.write_with_response = write_with_response
        self.connect_timeout = connect_timeout
        self.reconnect_delay = reconnect_delay
        self._logger = logger
        self._queue: "queue.Queue[Optional[bytes]]" = queue.Queue(maxsize=32)
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._last_log_at: dict[str, float] = {}

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        if self._stop_event.is_set():
            return

        self._thread = threading.Thread(
            target=self._thread_main,
            name=f"ble-gatt-writer-{self.address}",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass

    def write(self) -> bool:
        if self._stop_event.is_set():
            return False

        if self._thread is None or not self._thread.is_alive():
            self.start()

        try:
            self._queue.put_nowait(self.payload)
            print("1", flush=True)
            return True
        except queue.Full:
            self._log_throttled(
                "queue_full",
                "warning",
                "BLE write queue is full, dropping payload",
            )
            return False

    def _thread_main(self) -> None:
        try:
            asyncio.run(self._run())
        except Exception as exc:
            self._log("warning", f"BLE writer stopped: {exc}")

    async def _run(self) -> None:
        try:
            from bleak import BleakClient
        except Exception as exc:
            self._log("warning", f"BLE disabled, cannot import bleak: {exc}")
            self._stop_event.set()
            return

        client = BleakClient(self.address, timeout=self.connect_timeout)
        pending: Optional[bytes] = None

        try:
            while not self._stop_event.is_set():
                if not getattr(client, "is_connected", False):
                    if not await self._ensure_connected(client):
                        await asyncio.sleep(self.reconnect_delay)
                        continue

                if pending is None:
                    try:
                        pending = await asyncio.to_thread(
                            self._queue.get, True, 0.1
                        )
                    except queue.Empty:
                        continue

                if pending is None:
                    break

                if self._stop_event.is_set():
                    break

                try:
                    await client.write_gatt_char(
                        self.characteristic,
                        pending,
                        response=self.write_with_response,
                    )
                    pending = None
                except Exception as exc:
                    self._log_throttled(
                        "write_failed",
                        "warning",
                        f"BLE write failed: {exc}",
                    )
                    await self._disconnect(client)
                    await asyncio.sleep(self.reconnect_delay)
        finally:
            await self._disconnect(client)

    async def _ensure_connected(self, client: Any) -> bool:
        if getattr(client, "is_connected", False):
            return True

        try:
            await client.connect()
            self._log("info", f"BLE connected: {self.address}")
            return True
        except Exception as exc:
            self._log_throttled(
                "connect_failed",
                "warning",
                f"BLE connect failed for {self.address}: {exc}",
            )
            return False

    async def _disconnect(self, client: Any) -> None:
        if not getattr(client, "is_connected", False):
            return
        try:
            await client.disconnect()
            self._log("info", f"BLE disconnected: {self.address}")
        except Exception as exc:
            self._log_throttled(
                "disconnect_failed",
                "warning",
                f"BLE disconnect failed for {self.address}: {exc}",
            )

    def _log_throttled(
        self, key: str, level: str, message: str, interval: float = 5.0
    ) -> None:
        now = time.monotonic()
        if now - self._last_log_at.get(key, 0.0) < interval:
            return
        self._last_log_at[key] = now
        self._log(level, message)

    def _log(self, level: str, message: str) -> None:
        if self._logger is None:
            print(message)
            return
        try:
            self._logger(level, message)
        except Exception:
            print(message)


_SHARED_BLE_GATT_WRITERS: "weakref.WeakValueDictionary[tuple[str, str, bytes, bool], _BleGattWriter]" = (
    weakref.WeakValueDictionary()
)


class _BleFrameTrigger:
    def __init__(
        self,
        state_name: str,
        ble_mac: str = "",
        ble_mac_address: str = "",
        ble_characteristic: str = "",
        ble_characteristic_address: str = "",
        ble_characteristic_uuid: str = "",
        ble_trigger_frames: Optional[list] = None,
        ble_frames: Optional[list] = None,
        ble_write_byte: int = 1,
        ble_write_with_response: bool = False,
    ):
        self.state_name = state_name
        self.mac = str(ble_mac or ble_mac_address or "").strip()
        self.characteristic = str(
            ble_characteristic
            or ble_characteristic_address
            or ble_characteristic_uuid
            or ""
        ).strip()
        raw_frames = ble_trigger_frames if ble_trigger_frames is not None else ble_frames
        self.trigger_frames = self._parse_trigger_frames(raw_frames)
        self.write_payload = self._parse_write_payload(ble_write_byte)
        self.write_with_response = bool(ble_write_with_response)
        self._writer: Optional[_BleGattWriter] = None
        self._triggered_frames: set[int] = set()
        self._last_frame: Optional[int] = None

    def _writer_key(self) -> tuple[str, str, bytes, bool]:
        return (
            self.mac,
            self.characteristic,
            self.write_payload,
            self.write_with_response,
        )

    @staticmethod
    def _parse_trigger_frames(frames: Optional[list]) -> set[int]:
        if frames is None:
            return set()
        if isinstance(frames, str):
            frame_values = [
                value.strip() for value in frames.split(",") if value.strip()
            ]
        else:
            frame_values = frames
        return {int(float(frame)) for frame in frame_values}

    @staticmethod
    def _parse_write_payload(value: Any) -> bytes:
        if isinstance(value, (bytes, bytearray)):
            if not value:
                return b"\x01"
            return bytes([value[0] & 0xFF])
        if isinstance(value, str):
            number = int(value.strip(), 0)
        else:
            number = int(value)
        return bytes([number & 0xFF])

    def configured(self) -> bool:
        return bool(self.mac and self.characteristic)

    def start(self, ctx: BxiExample) -> None:
        if not self.configured():
            return
        if self._writer is None:
            key = self._writer_key()
            self._writer = _SHARED_BLE_GATT_WRITERS.get(key)
            if self._writer is None:
                self._writer = _BleGattWriter(
                    self.mac,
                    self.characteristic,
                    self.write_payload,
                    self.write_with_response,
                    logger=self._make_logger(ctx),
                )
                _SHARED_BLE_GATT_WRITERS[key] = self._writer
        self._writer.start()

    def stop(self) -> None:
        if self._writer is None:
            return
        self._writer.stop()
        self._writer = None

    def reset(self) -> None:
        self._triggered_frames.clear()
        self._last_frame = None

    def write_now(self, ctx: BxiExample) -> None:
        self.start(ctx)
        if self._writer is None:
            return
        self._writer.write()

    def write_for_timestep(
        self,
        ctx: BxiExample,
        timestep: float,
        trigger_frames: Optional[set[int]] = None,
    ) -> None:
        frames = self.trigger_frames if trigger_frames is None else trigger_frames
        if not frames:
            return

        self.start(ctx)
        if self._writer is None:
            return

        frame = int(timestep)
        if self._last_frame is None or frame <= self._last_frame:
            candidates = (frame,)
        else:
            candidates = range(self._last_frame + 1, frame + 1)

        for candidate in candidates:
            if candidate not in frames:
                continue
            if candidate in self._triggered_frames:
                continue
            self._writer.write()
            self._triggered_frames.add(candidate)

        self._last_frame = frame

    def _make_logger(self, ctx: BxiExample) -> Callable[[str, str], None]:
        def log(level: str, message: str) -> None:
            get_logger = getattr(ctx, "get_logger", None)
            if callable(get_logger):
                logger = get_logger()
                log_method = getattr(logger, level, None)
                if callable(log_method):
                    log_method(f"[{self.state_name}] {message}")
                    return
            print(f"[{self.state_name}] {message}")

        return log


class NormalState(RobotControlState):
    def on_prepare_enter(
        self,
        ctx: BxiExample,
        from_state: StateBehavior[BxiExample],
        transition: TransitionProfile,
    ) -> None:
        super().on_prepare_enter(ctx, from_state, transition)
        ctx.preheat_model(
            ctx.normal,
            with_cmd_vel=True,
            cmd_vel=self.get_cmd_vel(ctx),
        )

    def on_enter(self, ctx: BxiExample) -> None:
        self.shaketime = 0

    def get_first_frame(self, ctx: BxiExample) -> Optional[MotorFrame]:
        return self._motor_frame(
            ctx.normal.target_dof_pos, ctx.normal.kps, ctx.normal.kds
        )

    def get_motor_frame(
        self, ctx: BxiExample, dt: float, on_translation: bool
    ) -> Optional[MotorFrame]:
        cmd_vel = self.get_cmd_vel(ctx)
        q, dq = ctx.get_model_joint_state(ctx.normal)
        qpos, vel = ctx.normal.inference_step(
            q,
            dq,
            ctx.current_quat_wxyz,
            ctx.current_omega,
            cmd_vel,
        )
        frame = self._motor_frame(qpos, ctx.normal.kps, ctx.normal.kds)

        return frame

    def on_update(self, ctx: BxiExample, dt: float) -> None:
        if ctx.is_orientation_unsafe(ctx.current_quat_xyzw):
            print("check safe error, zero_torque!")
            ctx.request_state("zero_torque", trigger="safety")
            return

        frame = self.get_motor_frame(ctx, dt, False)
        if frame is not None:
            ctx.set_motor_target(*frame)


class ZeroTorqueState(RobotControlState):
    def get_first_frame(self, ctx: BxiExample) -> Optional[MotorFrame]:
        return self._motor_frame(
            ctx.joint_nominal_pos,
            np.zeros(ctx.dof_num, dtype=np.float32),
            np.zeros(ctx.dof_num, dtype=np.float32),
        )

    def get_motor_frame(
        self, ctx: BxiExample, dt: float, on_translation: bool
    ) -> Optional[MotorFrame]:
        frame = self._motor_frame(
            ctx.joint_nominal_pos,
            np.zeros(ctx.dof_num, dtype=np.float32),
            np.zeros(ctx.dof_num, dtype=np.float32),
        )
        return frame


class PdBrakeState(RobotControlState):
    def get_first_frame(self, ctx: BxiExample) -> Optional[MotorFrame]:
        return self._motor_frame(ctx.pd_pos, ctx.normal.kps, ctx.normal.kds)

    def get_motor_frame(
        self, ctx: BxiExample, dt: float, on_translation: bool
    ) -> Optional[MotorFrame]:
        frame = self._motor_frame(ctx.pd_pos, ctx.normal.kps, ctx.normal.kds)
        return frame


class InitialPosState(RobotControlState):
    def get_first_frame(self, ctx: BxiExample) -> Optional[MotorFrame]:
        return self._motor_frame(ctx.initial_pos, ctx.joint_kp, ctx.joint_kd)

    def get_motor_frame(
        self, ctx: BxiExample, dt: float, on_translation: bool
    ) -> Optional[MotorFrame]:
        frame = self._motor_frame(ctx.initial_pos, ctx.joint_kp, ctx.joint_kd)
        return frame


class DanceState(RobotControlState):
    default_segments = (
        (0, 1036),
        (1036, 1800),
        (1800, 2800),
        (2800, 3800),
        (3800, 4800),
        (4800, 5800),
    )

    def __init__(
        self,
        name: str,
        state_id: int,
        start_frame: int = 100,
        segments: Optional[list] = None,
        dance_segments: Optional[list] = None,
        segment_indices: Optional[list] = None,
        dance_segment_indices: Optional[list] = None,
        ble_mac: str = "",
        ble_mac_address: str = "",
        ble_characteristic: str = "",
        ble_characteristic_address: str = "",
        ble_characteristic_uuid: str = "",
        ble_trigger_frames: Optional[list] = None,
        ble_frames: Optional[list] = None,
        ble_write_byte: int = 1,
        ble_write_with_response: bool = False,
    ):
        super().__init__(name, state_id)
        self.start_frame = start_frame
        self.playing = True
        raw_segments = segments if segments is not None else dance_segments
        raw_indices = (
            segment_indices
            if segment_indices is not None
            else dance_segment_indices
        )
        self.segments = self._parse_segments(raw_segments)
        self.segment_indices = self._parse_segment_indices(raw_indices)
        self.segment_cursor = 0
        self._ble_frame_trigger = _BleFrameTrigger(
            name,
            ble_mac=ble_mac,
            ble_mac_address=ble_mac_address,
            ble_characteristic=ble_characteristic,
            ble_characteristic_address=ble_characteristic_address,
            ble_characteristic_uuid=ble_characteristic_uuid,
            ble_trigger_frames=ble_trigger_frames,
            ble_frames=ble_frames,
            ble_write_byte=ble_write_byte,
            ble_write_with_response=ble_write_with_response,
        )
        self.ble_mac = self._ble_frame_trigger.mac
        self.ble_characteristic = self._ble_frame_trigger.characteristic
        self.ble_trigger_frames = self._ble_frame_trigger.trigger_frames

    @classmethod
    def _parse_segments(cls, segments: Optional[list]) -> list[tuple[int, int]]:
        if segments is None:
            return list(cls.default_segments)
        if isinstance(segments, str):
            parts = [part.strip() for part in segments.split(";") if part.strip()]
            parsed_segments = []
            for part in parts:
                start_text, end_text = [value.strip() for value in part.split(",", 1)]
                parsed_segments.append((int(float(start_text)), int(float(end_text))))
            return cls._validate_segments(parsed_segments)

        parsed_segments = []
        for segment in segments:
            if isinstance(segment, str):
                start_text, end_text = [
                    value.strip() for value in segment.split(",", 1)
                ]
                start, end = int(float(start_text)), int(float(end_text))
            else:
                start, end = segment
                start, end = int(float(start)), int(float(end))
            parsed_segments.append((start, end))
        return cls._validate_segments(parsed_segments)

    @staticmethod
    def _validate_segments(segments: list[tuple[int, int]]) -> list[tuple[int, int]]:
        if not segments:
            raise ValueError("DanceState segments must not be empty")
        for start, end in segments:
            if start < 0 or end <= start:
                raise ValueError(
                    f"DanceState segment must satisfy 0 <= start < end: {(start, end)}"
                )
        return segments

    def _parse_segment_indices(self, indices: Optional[list]) -> list[int]:
        if indices is None:
            return list(range(len(self.segments)))
        if isinstance(indices, str):
            index_values = [
                value.strip() for value in indices.split(",") if value.strip()
            ]
        else:
            index_values = indices
        parsed_indices = [int(float(index)) for index in index_values]
        if not parsed_indices:
            raise ValueError("DanceState segment_indices must not be empty")
        for index in parsed_indices:
            if index < 0 or index >= len(self.segments):
                raise ValueError(
                    f"DanceState segment index out of range: {index}, "
                    f"segments={len(self.segments)}"
                )
        return parsed_indices

    def _current_segment(self) -> tuple[int, int]:
        return self.segments[self.segment_indices[self.segment_cursor]]

    def _reset_segment_playback(self, ctx: BxiExample) -> None:
        self.segment_cursor = 0
        ctx.dance.timestep = self._current_segment()[0]

    def _advance_segment_if_needed(self, ctx: BxiExample) -> bool:
        if self.segment_cursor >= len(self.segment_indices):
            return False

        _, end_frame = self._current_segment()
        if ctx.dance.timestep < end_frame:
            return True

        self.segment_cursor += 1
        if self.segment_cursor >= len(self.segment_indices):
            return False

        ctx.dance.timestep = self._current_segment()[0]
        self._ble_frame_trigger.reset()
        return True

    def on_bind(self, ctx):
        pass

    def on_prepare_enter(
        self,
        ctx: BxiExample,
        from_state: StateBehavior[BxiExample],
        transition: TransitionProfile,
    ) -> None:
        super().on_prepare_enter(ctx, from_state, transition)
        self._reset_segment_playback(ctx)
        if hasattr(ctx.dance, "timeinit"):
            ctx.dance.timeinit = 0.0
        ctx.preheat_model(ctx.dance)
        self._ble_frame_trigger.reset()
        self._ble_frame_trigger.start(ctx)

    def on_enter(self, ctx: BxiExample) -> None:
        self.playing = True
        self._reset_segment_playback(ctx)
        self._ble_frame_trigger.reset()

    def on_exit(self, ctx: BxiExample) -> None:
        super().on_exit(ctx)

    def get_first_frame(self, ctx: BxiExample) -> Optional[MotorFrame]:
        return self._motor_frame(
            ctx.dance.target_dof_pos,
            ctx.dance.kps,
            ctx.dance.kds,
        )

    def get_motor_frame(
        self, ctx: BxiExample, dt: float, on_translation: bool
    ) -> Optional[MotorFrame]:
        if not self._advance_segment_if_needed(ctx):
            return None
        if ctx.dance.timestep >= ctx.dance.motionpos.shape[0]:
            return None

        if self.playing and not on_translation:
            self._ble_frame_trigger.write_for_timestep(ctx, ctx.dance.timestep)

        q, dq = ctx.get_model_joint_state(ctx.dance)
        qpos = ctx.dance.inference_step(
            q,
            dq,
            ctx.current_quat_wxyz,
            ctx.current_omega,
        )

        if self.playing:
            ctx.dance.timestep += 50 * dt  # 模型动画是50hz播放的，dt是推理间隔

        frame = self._motor_frame(
            qpos,
            ctx.dance.kps,
            ctx.dance.kds,
        )
        return frame

    def on_update(self, ctx: BxiExample, dt: float) -> None:
        if (
            not self._advance_segment_if_needed(ctx)
            or ctx.dance.timestep >= ctx.dance.motionpos.shape[0]
        ):
            print("Motion replay finished, resetting simulation.")
            self._reset_segment_playback(ctx)
            ctx.request_state(
                "normal",
                trigger="motion_finished",
                transition={
                    "base": "dual_running_blend",
                    "duration": 0.5,
                    "data": {"run_from": False},
                },
            )
            return

        if ctx.is_orientation_unsafe(ctx.current_quat_xyzw):
            print("check safe error, zero_torque!")
            ctx.request_state("zero_torque", trigger="safety")
            return

        frame = self.get_motor_frame(ctx, dt, False)
        if frame is not None:
            ctx.set_motor_target(*frame)

    def on_action(self, ctx: BxiExample, action_name: str) -> bool:
        if action_name != "toggle_dance_pause":
            return False

        self.playing = not self.playing
        return True


class DancePlaylistState(RobotControlState):
    def __init__(
        self,
        name: str,
        state_id: int,
        playlist: list,
        ble_mac: str = "",
        ble_mac_address: str = "",
        ble_characteristic: str = "",
        ble_characteristic_address: str = "",
        ble_characteristic_uuid: str = "",
        ble_trigger_frames: Optional[list] = None,
        ble_frames: Optional[list] = None,
        ble_write_byte: int = 1,
        ble_write_with_response: bool = False,
        return_state: str = "normal",
        finish_trigger: str = "motion_finished",
        end_transition: Optional[dict] = None,
    ):
        super().__init__(name, state_id)
        self.playing = True
        self.playlist = self._parse_playlist(playlist)
        self.playlist_cursor = 0
        self.return_state = return_state
        self.finish_trigger = finish_trigger
        self.end_transition = end_transition or {
            "base": "dual_running_blend",
            "duration": 0.5,
            "data": {"run_from": False},
        }
        self._ble_frame_trigger = _BleFrameTrigger(
            name,
            ble_mac=ble_mac,
            ble_mac_address=ble_mac_address,
            ble_characteristic=ble_characteristic,
            ble_characteristic_address=ble_characteristic_address,
            ble_characteristic_uuid=ble_characteristic_uuid,
            ble_trigger_frames=ble_trigger_frames,
            ble_frames=ble_frames,
            ble_write_byte=ble_write_byte,
            ble_write_with_response=ble_write_with_response,
        )

    @staticmethod
    def _parse_playlist(playlist: list) -> list[dict[str, Any]]:
        if not playlist:
            raise ValueError("DancePlaylistState playlist must not be empty")

        parsed = []
        for index, item in enumerate(playlist):
            if isinstance(item, str):
                parsed.append(
                    {
                        "policy": item,
                        "start": None,
                        "end": None,
                        "ble_trigger_frames": None,
                    }
                )
                continue

            if isinstance(item, (list, tuple)):
                if len(item) != 3:
                    raise ValueError(
                        "DancePlaylistState list item must be [policy, start, end]: "
                        f"item {index}={item}"
                    )
                policy_name, start, end = item
                parsed.append(
                    {
                        "policy": str(policy_name),
                        "start": int(float(start)),
                        "end": int(float(end)),
                        "ble_trigger_frames": None,
                    }
                )
                continue

            if not isinstance(item, dict):
                raise ValueError(
                    f"DancePlaylistState playlist item {index} must be dict/list/string"
                )

            policy_name = (
                item.get("policy")
                or item.get("policy_attr")
                or item.get("model")
                or item.get("name")
            )
            if not policy_name:
                raise ValueError(
                    f"DancePlaylistState playlist item {index} must define policy"
                )

            segment = item.get("segment") or item.get("frames")
            if segment is not None:
                if len(segment) != 2:
                    raise ValueError(
                        f"DancePlaylistState segment must be [start, end]: {segment}"
                    )
                start, end = segment
            else:
                start = item.get("start_frame", item.get("start"))
                end = item.get("end_frame", item.get("end"))

            start = None if start is None else int(float(start))
            end = None if end is None else int(float(end))
            if start is not None and end is not None and end <= start:
                raise ValueError(
                    "DancePlaylistState segment must satisfy start < end: "
                    f"{(start, end)}"
                )

            raw_ble_frames = (
                item["ble_trigger_frames"]
                if "ble_trigger_frames" in item
                else item.get("ble_frames")
            )
            parsed.append(
                {
                    "policy": str(policy_name),
                    "start": start,
                    "end": end,
                    "ble_trigger_frames": (
                        None
                        if raw_ble_frames is None
                        else _BleFrameTrigger._parse_trigger_frames(raw_ble_frames)
                    ),
                }
            )

        return parsed

    def _current_item(self) -> dict[str, Any]:
        return self.playlist[self.playlist_cursor]

    def _current_policy_name(self) -> str:
        return self._current_item()["policy"]

    def _current_ble_trigger_frames(self) -> Optional[set[int]]:
        item_frames = self._current_item().get("ble_trigger_frames")
        if item_frames is None:
            return None
        return self._ble_frame_trigger.trigger_frames | item_frames

    def _current_policy(self, ctx: BxiExample) -> Any:
        policy_name = self._current_policy_name()
        if not hasattr(ctx, policy_name):
            raise AttributeError(
                f"DancePlaylistState policy '{policy_name}' is not loaded on ctx"
            )
        return getattr(ctx, policy_name)

    def _motion_frame_count(self, policy: Any) -> Optional[int]:
        motionpos = getattr(policy, "motionpos", None)
        if motionpos is not None:
            return int(motionpos.shape[0])
        motioninputpos = getattr(policy, "motioninputpos", None)
        if motioninputpos is not None:
            return int(motioninputpos.shape[0])
        return None

    def _current_bounds(self, policy: Any) -> tuple[int, int]:
        item = self._current_item()
        start = item["start"]
        if start is None:
            start = int(getattr(policy, "start_frame", 0))

        end = item["end"]
        if end is None:
            frame_count = self._motion_frame_count(policy)
            if frame_count is not None:
                end = frame_count
            else:
                end = int(getattr(policy, "end_frame")) + 1

        if start < 0 or end <= start:
            raise ValueError(
                f"DancePlaylistState invalid segment for {item['policy']}: "
                f"{(start, end)}"
            )
        return start, end

    def _activate_current_item(
        self,
        ctx: BxiExample,
        reset_runtime: bool = False,
    ) -> Any:
        policy = self._current_policy(ctx)
        start, _ = self._current_bounds(policy)
        policy.timestep = start
        if hasattr(policy, "timeinit"):
            policy.timeinit = 0.0
        if reset_runtime:
            self._reset_policy_runtime(policy)
        return policy

    def _reset_policy_runtime(self, policy: Any) -> None:
        action_buffer = getattr(policy, "action_buffer", None)
        if action_buffer is not None:
            try:
                action_buffer[...] = 0.0
            except TypeError:
                policy.action_buffer = np.zeros_like(action_buffer, dtype=np.float32)

        history_buffers = getattr(policy, "history_buffers", None)
        if isinstance(history_buffers, dict):
            history_buffers.clear()

        history_lengths = getattr(policy, "observation_history_lengths", None)
        if history_lengths is not None:
            policy.obs_history_len = max(1, max(int(float(v)) for v in history_lengths))

    def _reset_playlist(
        self,
        ctx: BxiExample,
        reset_runtime: bool = False,
    ) -> Any:
        self.playlist_cursor = 0
        return self._activate_current_item(ctx, reset_runtime=reset_runtime)

    def _advance_playlist_if_needed(self, ctx: BxiExample) -> bool:
        if self.playlist_cursor >= len(self.playlist):
            return False

        policy = self._current_policy(ctx)
        _, end_frame = self._current_bounds(policy)
        if policy.timestep < end_frame:
            return True

        self.playlist_cursor += 1
        if self.playlist_cursor >= len(self.playlist):
            return False

        self._activate_current_item(ctx, reset_runtime=True)
        self._ble_frame_trigger.reset()
        return True

    def on_bind(self, ctx):
        pass

    def on_prepare_enter(
        self,
        ctx: BxiExample,
        from_state: StateBehavior[BxiExample],
        transition: TransitionProfile,
    ) -> None:
        super().on_prepare_enter(ctx, from_state, transition)
        policy = self._reset_playlist(ctx, reset_runtime=True)
        ctx.preheat_model(policy)
        self._ble_frame_trigger.reset()
        self._ble_frame_trigger.start(ctx)

    def on_enter(self, ctx: BxiExample) -> None:
        self.playing = True
        self._reset_playlist(ctx, reset_runtime=True)
        self._ble_frame_trigger.reset()

    def on_transition_runtime_enter(
        self,
        ctx: BxiExample,
        transition: TransitionProfile,
    ) -> None:
        self.playing = True
        self._reset_playlist(ctx, reset_runtime=False)
        self._ble_frame_trigger.reset()

    def get_first_frame(self, ctx: BxiExample) -> Optional[MotorFrame]:
        policy = self._current_policy(ctx)
        qpos = getattr(policy, "target_dof_pos", None)
        if qpos is None:
            qpos = getattr(policy, "default_dof_pos", None)
        if qpos is None:
            return None
        return self._motor_frame(qpos, policy.kps, policy.kds)

    def get_transition_frame(
        self,
        ctx: BxiExample,
        role: str,
        transition: TransitionProfile,
    ) -> Optional[MotorFrame]:
        frame = self.get_motor_frame(ctx, float(getattr(ctx, "dt", 0.0)), True)
        if frame is None:
            return None

        if self.playing and self._transition_bool(
            transition,
            "advance_during_transition",
            False,
        ):
            policy = self._current_policy(ctx)
            policy.timestep += 50 * float(getattr(ctx, "dt", 0.0))

        return frame

    def get_motor_frame(
        self, ctx: BxiExample, dt: float, on_translation: bool
    ) -> Optional[MotorFrame]:
        if not self._advance_playlist_if_needed(ctx):
            return None

        policy = self._current_policy(ctx)
        frame_count = self._motion_frame_count(policy)
        if frame_count is not None and policy.timestep >= frame_count:
            return None

        if self.playing and not on_translation:
            self._ble_frame_trigger.write_for_timestep(
                ctx,
                policy.timestep,
                self._current_ble_trigger_frames(),
            )

        q, dq = ctx.get_model_joint_state(policy)
        qpos = policy.inference_step(
            q,
            dq,
            ctx.current_quat_wxyz,
            ctx.current_omega,
        )

        if self.playing and not on_translation:
            policy.timestep += 50 * dt

        return self._motor_frame(qpos, policy.kps, policy.kds)

    def on_update(self, ctx: BxiExample, dt: float) -> None:
        if not self._advance_playlist_if_needed(ctx):
            print("Dance playlist finished, resetting simulation.")
            self._reset_playlist(ctx)
            ctx.request_state(
                self.return_state,
                trigger=self.finish_trigger,
                transition=self.end_transition,
            )
            return

        if ctx.is_orientation_unsafe(ctx.current_quat_xyzw):
            print("check safe error, zero_torque!")
            ctx.request_state("zero_torque", trigger="safety")
            return

        frame = self.get_motor_frame(ctx, dt, False)
        if frame is not None:
            ctx.set_motor_target(*frame)

    def on_action(self, ctx: BxiExample, action_name: str) -> bool:
        if action_name != "toggle_dance_pause":
            return False

        self.playing = not self.playing
        return True


class MotionState(RobotControlState):
    policy_attr = ""
    finish_trigger = "flip_finished"
    end_frame_trim = 0
    end_transition = {}

    def __init__(self, name: str, state_id: int):
        super().__init__(name, state_id)
        self.playing = True

    def on_bind(self, ctx):
        pass

    def _policy(self, ctx: BxiExample) -> Any:
        return getattr(ctx, self.policy_attr)

    def on_enter_transition(self, ctx, from_state, progress, transition):
        policy = self._policy(ctx)
        policy.timestep = policy.start_frame
        return super().on_enter_transition(ctx, from_state, progress, transition)

    def on_prepare_enter(
        self,
        ctx: BxiExample,
        from_state: StateBehavior[BxiExample],
        transition: TransitionProfile,
    ) -> None:
        super().on_prepare_enter(ctx, from_state, transition)
        policy = self._policy(ctx)
        if hasattr(policy, "timeinit"):
            policy.timeinit = 0.0
        ctx.preheat_model(policy)

    def on_enter(self, ctx: BxiExample) -> None:
        self.playing = True
        policy = self._policy(ctx)
        policy.timestep = policy.start_frame
        if hasattr(policy, "timeinit"):
            policy.timeinit = 0.0

    def get_first_frame(self, ctx: BxiExample) -> Optional[MotorFrame]:
        policy = self._policy(ctx)
        qpos = getattr(policy, "target_dof_pos", None)
        if qpos is None:
            qpos = getattr(policy, "default_dof_pos", None)
        if qpos is None:
            return None
        return self._motor_frame(qpos, policy.kps, policy.kds)

    def get_motor_frame(
        self, ctx: BxiExample, dt: float, on_translation: bool
    ) -> Optional[MotorFrame]:
        policy = self._policy(ctx)
        q, dq = ctx.get_model_joint_state(policy)

        qpos = policy.inference_step(
            q,
            dq,
            ctx.current_quat_wxyz,
            ctx.current_omega,
        )

        if self.playing and not on_translation:
            policy.timestep += 50 * dt  # 模型动画是50hz播放的，dt是推理间隔

        frame = self._motor_frame(qpos, policy.kps, policy.kds)
        return frame

    def on_update(self, ctx: BxiExample, dt: float) -> None:
        policy = self._policy(ctx)

        frame = self.get_motor_frame(ctx, dt, False)
        if frame is not None:
            ctx.set_motor_target(*frame)

        if policy.timestep > policy.end_frame - self.end_frame_trim:
            print("Motion replay finished, resetting simulation.")
            ctx.request_state(
                "normal", trigger=self.finish_trigger, transition=self.end_transition
            )


class BackFlipState(MotionState):
    policy_attr = "back_flip"
    finish_trigger = "back_flip_finished"
    end_frame_trim = 30
    end_transition = {
        "base": "dual_running_blend",
        "duration": 0.45,
        "data": {
            "curve": "linear",
            "run_from": True,
        },  # 过渡的时候模型继续推理，同时推理下一个模型
    }


class ForwardFlipState(MotionState):
    policy_attr = "forward_flip"
    finish_trigger = "forward_flip_finished"
    end_frame_trim = 125
    end_transition = {
        "base": "dual_running_blend",
        "duration": 1.0,
        "data": {
            "curve": "smootherstep",
            "run_from": True,
        },  # 过渡的时候模型继续推理，同时推理下一个模型
    }


class HandPlayBackState(RobotControlState):
    start_frame = 0
    tail_trim_frames = 0
    return_time = 0.5
    file_name = "applause.pkl"

    def __init__(self, name, state_id):
        super().__init__(name, state_id)
        self.frame = 0.0
        self.applause_data, self.fps = self._load_applause_data()

    def _load_applause_data(self) -> tuple[np.ndarray, float]:
        data_path = os.path.join(
            get_package_share_path("bxi_example_py_elf3"),
            "data",
            self.file_name,
        )
        with open(data_path, "rb") as data_file:
            data = pickle.load(data_file)

        dof_pos = np.asarray(data["dof_pos"], dtype=np.float32)[:, -14:]
        start = min(self.start_frame, dof_pos.shape[0])
        end = max(start, dof_pos.shape[0] - self.tail_trim_frames)
        applause_data = dof_pos[start:end]
        if applause_data.shape[0] == 0:
            raise ValueError(
                f"HandPlayBack data is empty after frame trim: {data_path}"
            )

        return applause_data, float(data["fps"])

    def on_prepare_enter(
        self,
        ctx: BxiExample,
        from_state: StateBehavior[BxiExample],
        transition: TransitionProfile,
    ) -> None:
        super().on_prepare_enter(ctx, from_state, transition)
        ctx.preheat_model(
            ctx.withoutarm,
            with_cmd_vel=True,
            cmd_vel=self.get_cmd_vel(ctx),
        )

    def on_enter(self, ctx: BxiExample) -> None:
        self.frame = 0.0
        self.playing = True

    def get_first_frame(self, ctx: BxiExample) -> Optional[MotorFrame]:
        qpos = ctx.withoutarm.target_dof_pos.copy()
        qpos[-14:] = self.applause_data[0]
        return self._motor_frame(qpos, ctx.withoutarm.kps, ctx.withoutarm.kds)

    def get_motor_frame(self, ctx, dt, on_translation):
        cmd_vel = self.get_cmd_vel(ctx)
        q, dq = ctx.get_model_joint_state(ctx.withoutarm)
        qpos, vel = ctx.withoutarm.inference_step(
            q,
            dq,
            ctx.current_quat_wxyz,
            ctx.current_omega,
            cmd_vel,
        )
        if self.frame < self.applause_data.shape[0]:
            qpos[-14:] = self.applause_data[int(self.frame)]
        else:
            qpos[-14:] = self.applause_data[-1]
        if self.playing and not on_translation:
            self.frame += self.fps * dt
        frame = self._motor_frame(qpos, ctx.withoutarm.kps, ctx.withoutarm.kds)
        return frame

    def on_update(self, ctx: BxiExample, dt: float) -> None:
        if ctx.is_orientation_unsafe(ctx.current_quat_xyzw):
            ctx.request_state("zero_torque", trigger="safety")
            return
        if self.frame >= self.applause_data.shape[0]:
            ctx.request_state(
                "normal",
                trigger="applause_finished",
                transition={
                    "base": "dual_running_blend",
                    "duration": 1.0,
                },
            )
            return
        frame = self.get_motor_frame(ctx, dt, False)
        if frame is not None:
            ctx.set_motor_target(*frame)

    def on_action(self, ctx: BxiExample, action_name: str) -> bool:
        if action_name != "toggle_dance_pause":
            return False

        self.playing = not self.playing
        return True


class ApplauseState(HandPlayBackState):
    start_frame = 600
    tail_trim_frames = 600
    file_name = "isaaclab_model/applause.pkl"


class HelloState(RobotControlState):
    def __init__(self, name, state_id):
        super().__init__(name, state_id)

    def on_prepare_enter(
        self,
        ctx: BxiExample,
        from_state: StateBehavior[BxiExample],
        transition: TransitionProfile,
    ) -> None:
        super().on_prepare_enter(ctx, from_state, transition)
        ctx.preheat_model(
            ctx.withoutarm,
            with_cmd_vel=True,
            cmd_vel=self.get_cmd_vel(ctx),
        )

    def on_enter(self, ctx: BxiExample) -> None:
        self.playing = True
        self.shaketime = 0

    def get_first_frame(self, ctx: BxiExample) -> Optional[MotorFrame]:
        qpos = ctx.withoutarm.target_dof_pos.copy()
        qpos[22] = -0.9
        qpos[24] = 0.0
        qpos[25] = -0.3
        return self._motor_frame(qpos, ctx.withoutarm.kps, ctx.withoutarm.kds)

    def get_motor_frame(
        self, ctx: BxiExample, dt: float, on_translation: bool
    ) -> Optional[MotorFrame]:
        if self.shaketime < 50:
            self.kp = self.shaketime / 50 * ctx.withoutarm.kps
        cmd_vel = self.get_cmd_vel(ctx)
        q, dq = ctx.get_model_joint_state(ctx.withoutarm)
        qpos, vel = ctx.withoutarm.inference_step(
            q,
            dq,
            ctx.current_quat_wxyz,
            ctx.current_omega,
            cmd_vel,
        )
        qpos[22] = -0.9
        qpos[24] = math.sin(self.shaketime / 10) * 0.5
        qpos[25] = -0.3
        if self.playing:
            self.shaketime += 1
        frame = self._motor_frame(qpos, self.kp, ctx.withoutarm.kds)
        return frame

    def on_update(self, ctx: BxiExample, dt: float) -> None:
        if ctx.is_orientation_unsafe(ctx.current_quat_xyzw):
            ctx.request_state("zero_torque", trigger="safety")
            return
        frame = self.get_motor_frame(ctx, dt, False)
        if frame is not None:
            ctx.set_motor_target(*frame)

    def on_action(self, ctx: BxiExample, action_name: str) -> bool:
        if action_name != "toggle_dance_pause":
            return False

        self.playing = not self.playing
        return True


class AmpRunState(RobotControlState):
    def __init__(self, name: str, state_id: int):
        super().__init__(name, state_id)
        self.max_vel = 0.0
        self.pre_cmd_vel_run = np.array([0.0, 0.0, 0.0])
        self.cmd_vel_run = np.array([0.0, 0.0, 0.0])

    def on_prepare_enter(
        self,
        ctx: BxiExample,
        from_state: StateBehavior[BxiExample],
        transition: TransitionProfile,
    ) -> None:
        super().on_prepare_enter(ctx, from_state, transition)
        ctx.preheat_model(
            ctx.amp_run,
            with_cmd_vel=True,
            cmd_vel=self.get_cmd_vel(ctx),
        )

    def on_enter(self, ctx: BxiExample) -> None:
        self.max_vel = 0.0
        self.pre_cmd_vel_run = np.array([0.0, 0.0, 0.0])
        self.cmd_vel_run = np.array([0.0, 0.0, 0.0])

    def get_first_frame(self, ctx: BxiExample) -> Optional[MotorFrame]:
        return self._motor_frame(
            ctx.amp_run.target_dof_pos, ctx.amp_run.kps, ctx.amp_run.kds
        )

    def process_cmd_vel(
        self,
        ctx: BxiExample,
        cmd_vel: np.ndarray,
    ) -> Optional[np.ndarray]:
        self.cmd_vel_run[:2] = 0.98 * self.pre_cmd_vel_run[:2] + 0.02 * cmd_vel[:2]
        self.cmd_vel_run[2] = cmd_vel[2]
        self.pre_cmd_vel_run = self.cmd_vel_run.copy()
        return self.cmd_vel_run

    def get_motor_frame(
        self, ctx: BxiExample, dt: float, on_translation: bool
    ) -> Optional[MotorFrame]:
        cmd_vel = self.get_cmd_vel(ctx)
        q, dq = ctx.get_model_joint_state(ctx.amp_run)
        qpos, vel = ctx.amp_run.inference_step(
            q,
            dq,
            ctx.current_quat_wxyz,
            ctx.current_omega,
            cmd_vel,
        )

        if vel[0] > self.max_vel:
            self.max_vel = vel[0]
        if ctx.loop_count >= 100 + int(0.3 / ctx.dt):
            print(self.max_vel)
            ctx.loop_count = int(0.3 / ctx.dt)
            self.max_vel = 0.0

        frame = self._motor_frame(qpos, ctx.amp_run.kps, ctx.amp_run.kds)
        return frame

    def on_update(self, ctx: BxiExample, dt: float) -> None:
        if ctx.is_orientation_unsafe(ctx.current_quat_xyzw):
            print("check safe error, zero_torque!")
            ctx.request_state("zero_torque", trigger="safety")
            return

        frame = self.get_motor_frame(ctx, dt, False)
        if frame is not None:
            ctx.set_motor_target(*frame)


class NormalRunState(RobotControlState):
    def on_prepare_enter(
        self,
        ctx: BxiExample,
        from_state: StateBehavior[BxiExample],
        transition: TransitionProfile,
    ) -> None:
        super().on_prepare_enter(ctx, from_state, transition)
        ctx.preheat_model(
            ctx.normal_run,
            with_cmd_vel=True,
            cmd_vel=self.get_cmd_vel(ctx),
        )

    def on_enter(self, ctx: BxiExample) -> None:
        self.shaketime = 0
        if hasattr(ctx.normal_run, "action"):
            ctx.normal_run.action = np.zeros_like(ctx.normal_run.action)

    def get_first_frame(self, ctx: BxiExample) -> Optional[MotorFrame]:
        qpos = ctx.normal_run.default_joint_pos.copy()
        if hasattr(ctx.normal_run, "target_q"):
            qpos += ctx.normal_run.target_q
        return self._motor_frame(
            qpos,
            ctx.normal_run.joint_stiffness,
            ctx.normal_run.joint_damping,
        )

    def get_motor_frame(
        self, ctx: BxiExample, dt: float, on_translation: bool
    ) -> Optional[MotorFrame]:
        cmd_vel = self.get_cmd_vel(ctx)
        q, dq = ctx.get_model_joint_state(ctx.normal_run)
        qpos = ctx.normal_run.infer_step(
            q,
            dq,
            ctx.current_quat_xyzw,
            ctx.current_omega,
            cmd_vel,
        )
        frame = self._motor_frame(
            qpos,
            ctx.normal_run.joint_stiffness,
            ctx.normal_run.joint_damping,
        )

        frame.qpos[29] = math.sin(self.shaketime / 10) * 0.2  # neck_z_joint 位置
        frame.qpos[30] = math.sin(self.shaketime / 5) * 0.2  # neck_y_joint 位置

        frame.kp[29] = 20.0  # neck_z_joint kp
        frame.kp[30] = 20.0  # neck_y_joint kp

        frame.kd[29] = 1.0  # neck_z_joint kd
        frame.kd[30] = 1.0
        self.shaketime += 1

        return frame

    def on_update(self, ctx: BxiExample, dt: float) -> None:
        if ctx.is_orientation_unsafe(ctx.current_quat_xyzw):
            print("check safe error, zero_torque!")
            ctx.request_state("zero_torque", trigger="safety")
            return

        frame = self.get_motor_frame(ctx, dt, False)
        if frame is not None:
            ctx.set_motor_target(*frame)
