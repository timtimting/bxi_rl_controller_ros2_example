# ELF3 半身演示

本文档对应 `test_half` 分支的 `com.bxi.basic_actions` 与 `com.bxi.half_demo` 两个 Mod，包含零力矩、初始姿态、零位、鼓掌、悬挂跑步、双臂行程和 IMU 失联保护。跑步使用录制轨迹，行程测试按预设的双臂关节顺序运动；下半身不使用学习模型控制。

## 关节与模型

- 实机仅向已接入的 19 个关节发送命令：腰部 3 个、左右髋 `y` 各 1 个、双臂 14 个。默认额外电机禁用掩码为 `0x60007DF0`，包含双腿未接入的 10 个电机和头部 2 个电机。启动前仍须核对实际接线及硬件日志中的最终 `motor_disable` 值。
- `example_demo.launch.py` 加载 `elf3_half.xml`。仿真保留并控制头部 2 个关节，共 21 个驱动关节；双腿的网格、碰撞体和髋以下结构已移除。双侧髋 `y` 关节及其惯量保留为不可见驱动体，以兼容 19 关节控制接口。
- 跑步和行程测试只规划上述 19 个关节；仿真头部使用控制器的默认零位目标。原完整模型 `elf3.xml` 仍供独立的全自由度仿真入口使用。

## 构建

在仓库根目录执行：

```bash
source /opt/bxi/bxi_ros2_pkg/setup.bash
bash build.sh
source install/setup.bash
```

需要 ROS 2 Humble、BXI 的 `communication`/`mujoco`/`hardware_elf3` 包及仓库依赖。`build.sh` 在 RK3588 上会检查模型的 RKNN 构建产物；首次执行可能需要额外时间。

## 启动仿真

在已加载环境的终端运行：

```bash
ros2 launch bxi_example_py_elf3 example_demo.launch.py
```

另一个终端同样加载 `/opt/bxi/bxi_ros2_pkg/setup.bash` 和 `install/setup.bash`，再启动遥控器。使用手柄时运行：

```bash
ros2 launch remote_controller remote_controller.launch.py
```

使用键盘时改运行 `ros2 launch remote_controller remote_controller_keyboard.launch.py`。同一 ROS 域内不要同时启动多个遥控器发布者。

## 启动实机

**先将机器人可靠悬挂，确认周围无人、IMU 和遥控器工作正常，并核对电机禁用掩码。** 零力矩会使机器人失去支撑；跑步与行程测试不提供在线电机温度或碰撞监测。软件限位不能替代机械限位、硬件保护和现场急停。

在已加载环境的终端运行：

```bash
ros2 launch bxi_example_py_elf3 example_demo_hw.launch.py
```

硬件 launch 默认读取 `/opt/bxi/robot_config.yaml`，同时启动相机、`hardware_elf3` 和控制节点。可以通过 `robot_config_file:=...`、`enable_imu:=...`、`motor_disable_mask:=...` 覆盖启动参数；不要在未确认接线和安全条件前放开默认禁用位。若 `ros_elf_launch.service` 已启动遥控器，不要再开第二个遥控器实例。

此演示固定要求最终电机禁用掩码为 `0x60007DF0`，不接受通过启动参数放开更多电机。控制节点还会查询 `/hardware_elf3` 的 `hardware_config/motor_disable` 参数；读不到或不匹配时不执行 reset，不进入有力矩控制。运行中每 2 秒复核，超过 5 秒未能复核或掩码变化会锁定零力矩。首次控制前需连续收到 3 帧完整、有序且数值有效的 19 关节反馈。

## 按键流程

启动后默认处于**零力矩**。`pd_brake` 的界面名称是**初始姿态**，`initial_pos` 的界面名称是**零位模式**。

| 按键 | 当前状态 | 效果 |
| --- | --- | --- |
| RB+B | 零力矩、零位、鼓掌或测试中 | 进入初始姿态；测试中会先平滑回位 |
| LB+B | 初始姿态且到位 | 进入悬挂跑步 |
| X | 悬挂跑步中 | 受控停止，回到初始姿态 |
| RB+Y | 零力矩、初始姿态或测试中 | 进入零位模式；测试中会先平滑回位 |
| LB+Y | 零位且到位 | 进入双臂行程测试；测试中再次按下则受控停止并回零位 |
| RT+B | 初始姿态或零位 | 进入鼓掌 |
| X | 鼓掌中 | 暂停或继续鼓掌 |
| RB+A | 初始姿态、零位、鼓掌或测试中 | 切换零力矩；这是紧急退出，不执行平滑回位 |

推荐顺序：**RB+B -> 等初始姿态到位 -> LB+B** 进行跑步；**RB+Y -> 等零位到位 -> LB+Y** 进行行程测试。`LB+A` 和旧的 `LB+RB+B/Y` 不再用于半身演示。

## 保护与排查

- 进入跑步或行程测试时，已接入关节与目标姿态的最大偏差须不超过 10 度。未到位时，切换会被拒绝；先等待姿态稳定，不要连续按键。
- 测试运行时，关节反馈超过 0.2 秒未更新、控制周期间隔超过 50 ms 或命令超过软件限位，会请求零力矩并锁定本次测试。回位要求最大偏差不超过 5 度。实机 IMU 失联也会进入保护状态。
- 硬件控制路径还检查每个已接入关节的反馈出现时间及完整帧时间（0.2 秒）、持续超过 0.8 rad 达 0.75 秒的目标与反馈偏差。外部命令覆盖生效后，再校验最终 19 关节名单、有限数值、位置软限位、速度不超过 10 rad/s、力矩上限和 `kp/kd` 上限（500/20）。故障会立即发送一帧零增益、零力矩命令，并持续拒绝普通命令；故障排除后需重启控制节点。力矩上限取自半身仿真模型，须与实机驱动额定值核对。
- **驱动接口限制**：`ActuatorStates` 只有关节名及位置、速度、力矩等字段，没有原始 CAN 电机 ID、故障码或逐电机设备采样时间。这里的“逐电机更新时间”是各关节在 ROS 消息中出现的时间；驱动若反复发布缓存值，控制层无法识别。因此软件目前不能独立证明实际在线 ID 与接线一致，也不能监测硬件故障码。上线前需由 `hardware_elf3` 提供真实 ID、故障码和采样时间并接入此门槛。
- **通信看门狗尚未确认**：现有闭源 `hardware_elf3` 包未提供可核验的独立命令超时保护接口，不能把控制层的 0.2 秒反馈检查当成硬件看门狗。须由驱动供应方确认“命令停止后自动断力矩”的实现和超时时间，并在悬挂机械保护下断开控制命令实测；确认之前不得将此软件保护视为可独立保证实机安全。
- 若按键无效，先检查 `ros2 topic echo /motion_commands --once` 是否收到对应 `btn_*`，再看控制日志中的 `state transition` 或 `target is unavailable`。遥控器服务可用 `systemctl status ros_elf_launch.service` 检查。
- 若控制节点一直等待 IMU，检查 `ros2 topic echo /hardware/imu_data --once`、IMU 串口及硬件启动日志。保护锁定后不要强制重试动作，应先排除故障并重新启动控制程序。

Mod 内部的测试参数和行程顺序见 [半身演示 Mod 说明](src/bxi_example_py_elf3/mods/com.bxi.half_demo/README.zh-CN.md)。
