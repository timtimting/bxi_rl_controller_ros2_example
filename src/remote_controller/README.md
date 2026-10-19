## 按键说明

1. 按键映射对照图片：![手柄](./ps4_key_map.png "keymap") 
2. 其中AXIS为线性按键，箭头方向为正方向，值范围:-32767 - 32767，除扳机键L2/R2外初始值均为0  
3. 其中BT为常规按键，按下为1，松开为0
4. 方向键属于线性按键，但只有两个值，图示AXIS两端的键分别为最小和最大
5. L2/R2扳机键初始值为-32767，按下后线性增加到最大值，同时扳机键是一个复合按键，按下后同时触发一个常规按键，复合关系看图

## 多设备自动选择

`remote_controller` 将每个 `sources.<设备名>` 视为一个候选输入设备，并且严格只激活一个。候选设备按 `priority` 选择；高优先级设备连续可用 `promote_stable_ms` 后会自动抢占低优先级设备。

```yaml
inputs:
  selection:
    scan_interval_ms: 100       # 驱动可用性扫描周期
    promote_stable_ms: 500      # 高优先级设备稳定多久才允许抢占

sources:
  gamepad:
    type: joystick
    device: /dev/input/by-id/usb-...-joystick
    priority: 50
    ready_timeout_ms: 1000
    loss_timeout_ms: 300
    cooldown_ms: 1000
    signals: { ... }

  keyboard:
    type: keyboard
    priority: 0
    signals: { ... }
```

切换顺序固定为：停止旧驱动、清除旧设备所有信号、发布一次零运动命令、启动新驱动、等待新驱动声明就绪、再允许正常输入。切换期间会抑制所有 edge 输出；因此必须在切换后先松开、再按下 `start/stop` 等边沿按键，避免状态继承造成误触发。设备连续不可用超过 `loss_timeout_ms` 后会被断开，并在 `cooldown_ms` 内不参与再次选择。

`signals.*.timeout_ms` 是单个数据字段的过期保护，只适用于 UDP/TCP 等持续流式输入；它不用于手柄或 CRSF 的设备断连判断。默认手柄配置不设置它，断连统一由驱动的 `is_available()` 和 `loss_timeout_ms` 处理。

## 周期发布与控制端超时保护

默认 `outputs.publish_on_change: false`：每次 10 ms 定时器执行时都发布完整的
`motion_commands`，包括摇杆保持不动时；消息时间戳每次更新。目标频率为 100 Hz，
实际频率取决于系统调度。这样一次摇杆归零消息丢失后，后续零速度消息仍能更新控制输入。
不要将按键事件改为每帧触发；控制端仍按槽位变化解析边沿，重复状态不会重复执行动作。

`bxi_example_py_elf3_demo` 默认启用两个参数：

- `remote_timeout_sec: 0.3`：控制周期发现超过 300 ms 未接收有效遥控消息，清零遥控速度并丢弃待处理遥控事件。
- `remote_max_age_sec: 0.3`：具有非零 ROS 时间戳的消息，若比接收时 ROS 时钟旧或超前超过 300 ms，或时间戳未递增，则拒收，不刷新接收看门狗。非有限速度同样拒收。

超时后不自动切换动作状态；依赖遥控速度的状态会接收到零速度，固定动作等仍按自己的状态逻辑执行。
恢复时同步第一帧的按键状态，避免按住的旧按钮触发动作；需要再次松开、按下才触发。
零时间戳的旧发布者仍兼容，但只能检查接收间隔，无法判断源指令是否陈旧。
跨电脑部署应同步时钟，所有发布者都应持续发送最新状态。若显式设置 `publish_on_change: true`，
保持摇杆不动会触发控制端接收超时，不适用于默认看门狗配置。

硬件和仿真 demo launch 均接受 `remote_timeout_sec`、`remote_max_age_sec` 参数。
状态话题 `/hardware/state_machine_info` 或 `/simulation/state_machine_info` 的 JSON 新增
`remote_input`，包含接收年龄、上次接收时的源消息年龄、接收/拒收/超时计数及最后拒收原因。
`source_age_ms_at_receive` 包含源时间戳到回调执行的全部时间，不能单独作为 DDS 网络耗时。

遥控回调使用独立回调组，按键边沿解析使用独立锁，不等待控制计算持有的框架锁。
这些改动不能保证系统整体停顿时仍按时执行看门狗，也不替代硬件侧的命令超时保护。

`--driver <type>` 和 `--keyboard` 保留为调试过滤器：它们只允许相应类型的候选设备参与选择。正常部署不要传该参数，让 YAML 优先级生效。

## 运行时 driver DEBUG

直接运行节点时添加 `--DEBUG`（也兼容小写 `--debug`），会每秒调用一次每个 input driver 的
`debug()` 并输出诊断信息。CRSF 的诊断包含最近一帧 CRC 正确的 16 路归一化通道值：

```bash
ros2 run remote_controller remote_controller --DEBUG --config /path/to/xbox_default.yaml
```

通过 launch 启动时，ROS 2 的自定义 launch 参数语法是 `DEBUG:=true`：

```bash
ros2 launch remote_controller remote_controller.launch.py DEBUG:=true
```

`ros2 launch --debug` 是 ROS 2 launch 系统自身的调试选项，不会传递给 input driver。

### 添加编译内置的新驱动

新驱动通常继承 `InputDriverBase`（或直接实现 `InputDriver`），并用 `register_input_driver_factory("类型名", factory)` 注册。`InputDriverBase::set_signal()` 已处理与 mapper 的同步。`is_available()` 必须非阻塞；驱动可按自身协议定义可用性。对 CRSF，建议仅在设备打开且近期收到校验通过的完整通道帧时返回可用。`is_ready()` 只有在已收到驱动认为足够安全的完整状态快照时才返回 `true`。

`crsf` 已是内置 driver。它兼容标准 CRC 正确的 CRSF `0x16` 通道帧地址 `0xC8` 和
高频头转接器 `usb_f2` 使用的 `0xEE`。driver 会继续暴露 `crsf.channel.1` 到
`crsf.channel.16` 作为调试/兼容 raw channel，同时按 `usb_f2` 的 ELRS Hybrid/Wide
低延迟协议从 CH1..CH4 的 10-bit 字里还原 Xbox 摇杆和 14 个按钮，输出
`crsf.xbox.*` 原始信号。

默认配置已经包含完整的 `crsf` 候选项；最小配置如下：

```yaml
sources:
  crsf:
    type: crsf
    device: /dev/ttyCRSF
    priority: 100
    ready_timeout_ms: 1000
    loss_timeout_ms: 300
    cooldown_ms: 1000
    baud_rate: 460800
    signals:
      crsf.left_x:        {from: crsf.xbox.left_x}
      crsf.left_y:        {from: crsf.xbox.left_y}
      crsf.right_x:       {from: crsf.xbox.right_x}
      crsf.right_y:       {from: crsf.xbox.right_y}
      crsf.trigger_left:  {from: crsf.xbox.trigger_left}
      crsf.trigger_right: {from: crsf.xbox.trigger_right}
      crsf.a:             {from: crsf.xbox.button.a}
      crsf.b:             {from: crsf.xbox.button.b}
      crsf.x:             {from: crsf.xbox.button.x}
      crsf.y:             {from: crsf.xbox.button.y}
      crsf.lb:            {from: crsf.xbox.button.lb}
      crsf.rb:            {from: crsf.xbox.button.rb}
      crsf.back:          {from: crsf.xbox.button.back}
      crsf.start:         {from: crsf.xbox.button.start}
      crsf.dpad_left:     {from: crsf.xbox.button.dpad_left}
      crsf.dpad_right:    {from: crsf.xbox.button.dpad_right}
```

`usb_f2` 协议约定：

- CH1：左摇杆 X 7 bit + A/B/X 3 bit；
- CH2：左摇杆 Y 7 bit + Y/LB/RB 3 bit；
- CH3：右摇杆 X 6 bit + Back/Start/LStick/RStick 4 bit；
- CH4：右摇杆 Y 6 bit + D-pad Up/Down/Left/Right 4 bit；
- CH5：固定 low，避免 ELRS arm；
- CH6/CH7：LT/RT。

按钮按位同时传输，所以 A/B/X/Y/LB/RB/Start/Back 等组合键不会再被 CH8/CH9 的单一离散值互斥掉。
默认配置中 Start 对应 stop，RStick 对应 start；D-pad 左/右也映射为 CRSF yaw 的 `+1/-1`。
`crsf.channel.1` 到 `crsf.channel.16` 默认按接收机范围 `172..1811` 归一化至 `[-1, 1]`。
未激活时
driver 做非阻塞协议 probe，只有近期收到 CRC 正确帧才会参与抢占；激活后停止收到有效帧
超过 `loss_timeout_ms` 就会断连。尚未编译的其他 `type` 不会导致节点启动失败：节点会记录
warning、忽略该候选项，并自动使用下一个可用设备；没有任何可用设备时保持安全停止。

## USB 到 demo 的新鲜度诊断

默认新增 `remote_controller/freshness`（`std_msgs/String`，10Hz JSON）诊断话题，不改变
`motion_commands` 接口、发布模式、输入选择、滤波或既有超时保护。正常时只有首次状态信息；
异常变化立即报告，同一异常持续时每5秒一行，恢复后报告一次。demo 的诊断回调组独立于遥控
命令回调，诊断打印不会占住遥控输入锁。原有 `communication_trace` 继续诊断执行器/控制线程阻塞。

```bash
ros2 topic echo /remote_controller/freshness --once
```

主要证据：

- `usb_read_age_ms`/`reads`/`bytes`：驱动实际 `read()` 读到数据的时间和累计数量；读取轮询成功不算新数据。
- `valid_frame_age_ms`/`sample_seq`/`sample_hz`：有效输入帧的时间、序号与最近约100ms的频率；内容相同的连续帧也计数。
- `crc_errors`：CRSF CRC 校验失败累计；收到坏帧不会刷新有效帧时间。
- `reader_age_ms`：读取线程最近一次执行；可以区分线程停顿与设备没有数据。
- `published`/`publish_age_ms`：`motion_commands` 发布调用返回的计数和年龄；不是下游接收确认。
- `timer_gaps`：10ms 发布定时器间隔超过100ms的累计次数；恢复后也保留，帮助判断进程调度/阻塞。
- `matched_subscribers`：当前 ROS 匹配订阅者数量；从非零掉到零时单独记录告警。
- demo 的 `remote_path` 摘要另列 `demo_age_ms`；使用已有有效命令接收看门狗数据。
- `instance` 区分发布进程重启，`io_kind` 说明测量层，`observability=host_arrival_only` 说明观测边界。
  计数及年龄使用单调时钟；诊断消息另外携带 ROS 时间戳供 demo 检查陈旧/倒序消息。

| 状态 | 可以定位到的环节 |
| --- | --- |
| `usb_stale` | 周期源没有新的 USB/串口读取数据 |
| `protocol_stale` | 字节仍到达，但没有新的 CRC 正确通道帧；结合 `crc_errors` 判断 |
| `reader_stalled` | 驱动读取循环未继续执行 |
| `device_closed` | 设备描述符已关闭；`last_stop_status` 另存设备停用前的状态 |
| `publisher_stale` | 驱动诊断仍到达，但 remote controller 未及时发布命令 |
| `demo_receive_stale` | 发布端报告持续发布，demo 没有新的有效命令；应继续查 DDS/回调/消息校验 |
| `status_link_stale` | 诊断话题自身停更；不能仅凭此推断 USB 已断开 |
| `valid_frames_not_advancing` | 诊断消息仍刷新，但有效帧序号没有推进 |
| `event_source_unverified` | 非周期输入源，静止无事件正常，无法用空闲时长证明 USB 数据过期 |

CRSF（通常通过 `/dev/ttyCRSF` 的 USB 串口）按完整有效通道帧检测，阈值使用该源的
`loss_timeout_ms`。帧没有设备时间戳/无线链路序号，因此这里只能证明主机读到了新帧，
不能证明接收机内部没有重发旧数据，也不能测出无线端到主机的绝对延迟。

周期源在设备关闭、读取线程停顿、有效帧或 USB 报告超过 `loss_timeout_ms` 时，
`remote_controller` 会停止当前输入候选、清空该源所有映射值并立即走零命令安全屏障。
demo 若收到明确过期的源端诊断，也会清空速度/按键并拒收命令；恢复的首帧只同步按键，
不重放按住期间的动作。诊断消息自身超时为500ms；从未收到诊断的旧版发布者仍按
原来的300ms命令接收看门狗保护。进程整体卡死时软件无法保证及时执行清空，
硬件通信看门狗仍然必需。
`publish_on_change: true` 的静止无发布会标为 `on_change_idle_or_fresh`，不会被这套诊断
误报成 USB 故障；demo 原有的命令接收看门狗仍独立工作。

Linux joystick `/dev/input/js*` 通常只报告变化事件。驱动会尝试从该设备的 sysfs 路径
找到唯一匹配的 `/dev/hidraw*`，独立只读打开，记录原始 HID 输入报告，不抢占设备、
不消费 joystick 事件队列。若权限不足或无法唯一匹配，会明确提示新鲜度不可观测，
仍正常使用原 joystick 驱动，绝不把轮询空转伪装成新 USB 数据。可在 `sources.gamepad` 配置：

```yaml
# 可选：自动发现失败时指定与该手柄匹配的原始设备
freshness_hidraw_device: /dev/hidraw3
# 默认 false：只有确认设备在摇杆静止时也持续发送报告后，才设为 true
freshness_hidraw_periodic: true
```

`io_kind=usb_hidraw_report` 表示确实读到了原始 HID 报告；未取得 hidraw 时为
`joystick_event`，不能把事件静止当作 USB 断链。观察者按原读取循环轮询，最长正常
轮询间隔约100ms，报告年龄是用户态读取时间，不是 USB 硬件采样时间。
只有实际确认静止也持续出报告，才能启用 `freshness_hidraw_periodic: true`；启用后
hidraw 不可读会阻止输入就绪，运行中断开会清空输出。蓝牙手柄如果只在状态变化时
发送报告，无法单靠 `/dev/input/js*` 证明链路实时性，应继续依赖连接事件和命令看门狗。

部署时一起构建 `remote_controller` 与 `bxi_example_py_elf3`，重新加载 install 环境并重启
这两个节点。只更新 demo 无法获得源端证据，会显示 `status_unavailable`。
