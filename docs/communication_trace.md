# 通信停顿的内置记录

用于 ELF3 控制节点与硬件节点运行中突然停顿的定位。属于平台公共能力，不依赖某个 Mod。
通信记录本身只采证据；遥控源的新鲜度保护另外会在明确过期时清空输入。

## 启用与部署

控制节点默认启用，无需独立诊断脚本。监测线程每 50 ms 检查一次。已开始收发的
电机命令、关节反馈和 IMU 超过 200 ms 未更新会告警；遥控输入阈值与控制端的
`remote_timeout_sec` 相同，默认 300 ms。尚未收到第一帧的数据流不会报停更。

控制端输出按异常变化收敛：

- 首次异常在主日志输出一行短摘要，完整 `communication_trace` JSON 按 `diagnostic_id`
  写入启动时打印的 `bxi_communication_trace_<pid>_<时间>.jsonl`。
- 同一异常持续时，每 5 秒只输出一行摘要，不重复端点/QoS、压力、线程栈和历史计数。
  遥控输入反复短时停更也合并为一轮；连续 5 秒稳定后才提示恢复。
  此时不再查询 ROS 图；重复的同类时间戳异常也会合并。
- 新增停更数据流、阻塞阶段或异常类型时，立即保留新的完整现场，不等待 5 秒。
- 非遥控停更恢复时立即提示；原始旧、新时间戳保留在 JSON 中，主日志不再重复长数字。
  完全恢复后，再次停更仍会输出新的完整诊断。

例如，只有遥控输入停止更新时，持续告警类似：

```text
[WARN] [fw.runtime]: 通信告警#1: 遥控输入停更6.49s；状态=zero_torque；频率 电机命令=50.0Hz, 关节反馈=883.0Hz, IMU=475.0Hz, 遥控输入=0.0Hz；遥控有效=否
```

频率根据最近约 500 ms 的计数估算，启动时窗口可能更短；`--` 表示尚未观测。
“停更”指距上次回调入口/发布返回的时间，不等于消息的传输耗时。机器人没有操作时，
如果遥控发布端按变化发送，也可能出现这种记录；不能仅凭它认定 ROS 故障。
主日志精简不改变原有 300 ms 命令接收超时；新增的源端过期门控见下文。
硬件端原有日志策略不受此修改影响。
控制端 JSON 文件位于 `rclpy.logging.get_logging_directory()` 返回的 ROS 日志目录，
每个文件限制为 16 MiB，轮转备份为同名 `.1`。目录不可写时退回 ROS 日志中的完整 JSON。

硬件端需要同时安装对应修改。补丁 [communication_trace_hardware.patch](communication_trace_hardware.patch)
基于 `bxi_ros2_pkg_src` 提交 `74a2a92`，涉及 `hardware_elf3` 的两个既有文件及一个新增头文件。
在硬件源码仓库根目录检查并应用补丁，再构建 `communication` 和 `hardware_elf3`：

```bash
git apply --check /path/to/bxi_controller_ros2/docs/communication_trace_hardware.patch
git apply /path/to/bxi_controller_ros2/docs/communication_trace_hardware.patch
colcon build --packages-select communication hardware_elf3 --cmake-args -DCMAKE_CXX_STANDARD=17
```

若补丁已应用，跳过 `git apply`。本机 `/opt/bxi/bxi_ros2_pkg` 的旧 `communication` 没有
CANFD 消息；编译当前硬件源码需要源码版本的接口包，不能混用旧头文件。
构建控制器使用项目原有的 `colcon build --packages-select bxi_example_py_elf3 --symlink-install --merge-install`。

启动前加载控制器环境，并最后加载硬件工作区的 `install/local_setup.bash`，使新硬件包覆盖
`/opt/bxi` 的旧包。用 `ros2 pkg prefix hardware_elf3` 核对实际选择的位置。
编译源文件不会自动替换 `/opt/bxi` 中正在使用的二进制。

控制端启动时应看到 `communication trace details: ...jsonl`，硬件端仍输出
`communication_trace` / `"event":"started"`。原来的启动命令无需增加其他进程：

```bash
ros2 launch bxi_example_py_elf3 example_demo_hw.launch.py enable_imu:=false \
  > /var/log/bxi_log/$(date +%Y-%m-%d_%H-%M-%S)_elf.log 2>&1 &
```

直接运行时该重定向仍能保存两个节点输出。通过遥控器 `system.start` 启动时，
改由 `start_elf_logged.sh` 把输出分成同一时间戳的两份：

- `/var/log/bxi_log/<时间>_elf.log`：主日志，保留原始普通日志、短摘要和异常。
- `/var/log/bxi_log/trace/<时间>_elf.trace.log`：硬件端完整 `communication_trace` JSON。

实时查看最新主日志可使用安装后的彩色终端查看器：

```bash
"$(ros2 pkg prefix remote_controller)/share/remote_controller/scripts/view_elf_log.py"
```

它按日志等级及状态切换、遥控、IMU、硬件通信标记着色，`Ctrl+C` 退出；
服务重启产生新日志文件后重新运行即可切到新文件。
`--once` 只显示最近 80 行，`--color never` 禁用颜色。颜色只作用于终端，
`start_elf_logged.sh` 写入的日志保持纯文本。直接在终端运行 `ros2 launch` 时，
可在命令前设置 `RCUTILS_COLORIZED_OUTPUT=1` 启用 ROS 2 自带的等级颜色；
不要给写文件的自启动服务全局设置该变量。

控制端的完整 JSON 始终另存于启动日志打印的 `bxi_communication_trace_*.jsonl`。

硬件端重复相同的 trace 告警在主日志每5秒最多一行，完整内容仍逐条保留在 trace 文件。
日志目录不可写时输出回退到启动进程的 journal，避免因分流器退出造成 ROS stdout 堵塞。
跨电脑启动时，需要同时保留两端的日志。

## 记录内容

控制侧记录：

- `command_tx`：发布调用返回的次数、最后一帧时间戳和距今时间；不代表硬件已经收到。
- `joint_rx`、`imu_rx`、`remote_rx`：业务回调入口次数、消息时间戳、消息年龄和 DDS 时间信息。
- `active.control.stage`：`input_snapshot`、`framework_lock`、`state_update`、
  `command_publish` 或 `waiting_next_cycle`；记录卡住时的阶段，不必等待该周期结束。
- 活跃回调阶段、异常开始时的 Python 栈位置、Linux 线程 ID、CPU 亲和性、实际调度策略与优先级。
- 最近约 500 ms 的计数和控制阶段、ROS Domain、实际 RMW、`use_sim_time`、
  端点节点名/GID/QoS、命令发布端匹配的订阅数、系统负载与 CPU/IO/内存压力。
- 遥控新鲜度统计，以及正常周期结束后原有时序日志中的四阶段耗时。

硬件侧记录：

- `command_rx`：命令进入回调；`command_applied`：通过现有名称检查并写入命令缓冲；
  不表示电机执行，`step=0` 时原有逻辑仍将命令置零。
- `joint_tx`：反馈发布调用返回；`motor_tx`：`motor_cmd_trans` 返回，附最近应用的命令时间戳；
  不表示电机已反馈。`motor_wait_result` 和 `motor_timeouts` 记录底层反馈等待结果。
- `joint_lane.stage`：`motor_feedback_wait`、`state_read_and_kinematics`、
  `feedback_publish`、`command_lock_and_kinematics`、`motor_cmd_trans`、
  `user_canfd_io` 或 `waiting_executor`；另记录命令回调阶段。
- DDS 原始时间、匹配端点数量、旧命令保护次数、被现有检查拒绝的命令数量及当前 `step`。
- 单调时间、系统时间、监测线程自身的停顿及系统时钟跳变。

不同主机的 `mono_ns` 不可直接相减。相同主机也应核对启动周期；控制端记录 `boot_id`。
以消息的原始 `stamp_ns`、方向、节点和时间窗口关联两端记录。
这些是并发采样的诊断快照，不是一个原子化的全系统快照。

## 如何判断故障层级

| 同次故障的证据 | 定位方向 |
| --- | --- |
| 控制 `command_tx` 停止，`state_update` 持续很久 | 状态计算、推理或其内部阻塞；看栈和周期阶段耗时 |
| 控制停在 `framework_lock` / `input_snapshot` | 框架锁或输入路径阻塞；看活跃回调栈 |
| 控制停在 `command_publish` | 消息转换、覆盖计算或 ROS 发布调用；看栈 |
| 控制停在 `waiting_next_cycle`，出现 `watchdog_gap` | 调度/GIL/进程整体停顿的嫌疑；结合线程调度信息与系统压力 |
| 接收端 `dds_to_callback_ms` 明显增大 | 数据在 DDS 接收至业务回调入口之间等待；包括执行器、回调组、GIL、调度、take/反序列化等 |
| `source_to_dds_ms` 明显增大，而回调等待小 | 检查发布端中间件、DDS/共享内存/网络路径及两端时钟；不是纯网络耗时证明 |
| 控制持续发布，硬件命令回调停止，同时硬件 `joint_lane` 长时间卡在某阶段 | 硬件执行器被该阶段阻塞，需看相应底层函数 |
| 命令进入并应用，`motor_tx` 停止 | 命令到达后的底层处理/发送停顿 |
| `motor_tx` 持续，但 `motor_timeouts` 上升 | 底层电机反馈路径；不能再单凭现象归因于 ROS |
| 命令继续接收，但 `watchdog_drops` 增加、`step=0` | 旧命令保护已触发，当前零输出可能是保护结果 |
| 匹配端点变为零，或日志有不兼容 QoS 警告 | 进程退出、发现、Domain、QoS 或端点连接问题 |
| `wall_clock_jump`，或命令年龄与单调接收间隔明显矛盾 | 系统/ROS 时钟问题，检查 NTP、手动校时和 `use_sim_time` |

## 能力边界

- DDS 时间字段由 RMW 提供，[字段定义](https://raw.githubusercontent.com/ros2/rmw/humble/rmw/include/rmw/types.h)
  说明取样位置取决于实现；`null` 表示不支持或未提供，不能解释成零延迟。
  跨主机时 `source_to_dds_ms` 还包含时钟偏差；同机时钟跳变也会污染它和 `dds_to_callback_ms`。
- ROS 图中发现端点不等于 QoS 匹配，匹配不等于消息交付。Best Effort 的漏帧不能仅凭回调计数
  确定丢在哪一层；需要结合双方时间戳。如果 RMW 缺少接收时间，就不能确定消息到达 DDS 的时间。
- CPython 监测线程仍受 GIL 和系统调度约束。永久持有 GIL、系统挂死、进程被直接杀死或日志写入永久
  阻塞时，无法保证现场及时输出；恢复后可能只看到监测线程空窗。不能保证所有故障一次精确到某行代码。
- 硬件只记录底层函数阶段和返回结果，不能代替 SDK 内部或 CAN 总线抓包。
- `CommunicationExecutor` 适配本项目 ROS 2 Humble 的两个执行器内部钩子；升级 ROS/rclpy 后需重新验证。

## 已执行的验证

控制侧故障注入覆盖正常静默、进行中的控制停顿、瞬时停顿、恢复、时钟跳变、日志限频和有界缓存，
并验证业务回调异常仍正常传播。隔离 ROS 实测中，人为停止执行器约 450 ms 后，
`dds_to_callback_ms` 约 451 ms，`source_to_dds_ms` 小于 0.01 ms；没有向电机话题发送消息。
硬件补丁已在本机 x86 编译，并执行了原子采样/阶段退出/元数据记录的离线 C++ 检查。
尚未在目标 ARM 机器编译或在真机故障中验证。

## 遥控源端证据

`remote_controller` 现在发布低频 `remote_controller/freshness`，demo 的 `remote_path` 诊断
将 USB 字节读取、有效协议帧、命令发布、有效命令接收分开观察。源端确认超时后清空输入并发零命令；
demo 收到明确的源端故障时也会丢弃待处理输入。它与上面的执行器/控制线程
记录互补。重复源端异常每5秒一行；完全没有新诊断话题时只警告一次，正常不逐帧打印；USB 协议没有源时间戳时
明确标为仅能观测主机到达时间。CRSF 坏 CRC 不会刷新有效帧时间；同值但真实到达的新帧
会推进序号。普通 joystick 无事件不等于 USB 断链，只有匹配 hidraw 且已确认周期报告时
才启用周期源过期判断。字段、状态解释和可选配置见 `src/remote_controller/README.md`。
