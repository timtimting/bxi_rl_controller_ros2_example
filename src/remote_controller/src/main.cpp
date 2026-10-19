#include <chrono>
#include <cstdlib>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "remote_controller/config.hpp"
#include "remote_controller/input_device_manager.hpp"
#include "remote_controller/input_mapper.hpp"

using namespace std::chrono_literals;
using remote_controller::InputMapper;
using remote_controller::RemoteConfig;

class COMPublisher : public rclcpp::Node {
public:
    COMPublisher(
        const std::string &config_path,
        const std::string &driver_filter,
        bool driver_debug_enabled)
        : Node("COM_publisher"),
          mapper_(remote_controller::load_remote_config(config_path))
    {
        print_config_diagnostics(mapper_.config());
        com_pub_ = this->create_publisher<communication::msg::MotionCommands>(
            "motion_commands", 20);
        timer_ = this->create_wall_timer(10ms, std::bind(&COMPublisher::timer_callback, this));
        RCLCPP_INFO(
            this->get_logger(), "motion_commands publishing: %s (10 ms timer)",
            mapper_.config().publish_on_change ? "on change" : "periodic full state");

        input_manager_.reset(new remote_controller::InputDeviceManager(
            mapper_.config(),
            mapper_,
            lock_,
            [this](const std::vector<std::string> &outputs) { dispatch_outputs(outputs); },
            [this](const std::string &message) {
                RCLCPP_INFO(this->get_logger(), "%s", message.c_str());
            },
            driver_debug_enabled,
            driver_filter));
        freshness_pub_ = this->create_publisher<std_msgs::msg::String>("remote_controller/freshness", 10);
        freshness_timer_ = this->create_wall_timer(100ms, std::bind(&COMPublisher::freshness_callback, this));
        RCLCPP_INFO(this->get_logger(), "remote freshness: USB reads / valid frames / publish counters on remote_controller/freshness (10Hz); diagnostics only");
        if (driver_debug_enabled) {
            RCLCPP_INFO(this->get_logger(), "input driver debug is enabled");
        }
    }

    ~COMPublisher()
    {
        if (input_manager_) {
            input_manager_->stop();
        }
    }

private:
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr freshness_pub_;
    rclcpp::TimerBase::SharedPtr freshness_timer_;
    std::uint64_t published_count_ = 0;
    std::int64_t last_publish_ns_ = 0;
    const std::int64_t freshness_instance_ = remote_controller::freshness_now_ns();
    std::int64_t next_freshness_log_ns_ = 0;
    std::string last_freshness_status_, last_rate_source_;
    std::int64_t last_rate_ns_ = 0;
    std::uint64_t last_samples_ = 0, last_reads_ = 0;
    std::int64_t last_timer_ns_ = 0, next_timer_gap_log_ns_ = 0;
    std::uint64_t timer_gaps_ = 0;
    bool had_subscriber_ = false, subscriber_lost_ = false;

    void freshness_callback()
    {
        const auto s = input_manager_->input_freshness();
        const auto now = remote_controller::freshness_now_ns();
        const auto status = remote_controller::freshness_status(s, now);
        double sample_hz = -1, read_hz = -1;
        const std::string rate_source = s.source + ":" + s.io_kind;
        if (last_rate_ns_ && rate_source == last_rate_source_ && now > last_rate_ns_ && s.samples >= last_samples_ && s.reads >= last_reads_) {
            const double seconds = (now - last_rate_ns_) / 1e9;
            sample_hz = (s.samples - last_samples_) / seconds;
            read_hz = (s.reads - last_reads_) / seconds;
        }
        last_rate_source_ = rate_source; last_rate_ns_ = now; last_samples_ = s.samples; last_reads_ = s.reads;
        std_msgs::msg::String message;
        const auto subscribers = com_pub_->get_subscription_count();
        message.data = remote_controller::freshness_json(s, now, this->now().nanoseconds(),
            published_count_, last_publish_ns_, mapper_.config().publish_on_change,
            freshness_instance_, sample_hz, read_hz, timer_gaps_, subscribers);
        freshness_pub_->publish(message);
        if (subscribers > 0) {
            had_subscriber_ = true;
            if (subscriber_lost_) RCLCPP_INFO(this->get_logger(), "motion_commands ROS subscriber restored: %zu", subscribers);
            subscriber_lost_ = false;
        } else if (had_subscriber_ && !subscriber_lost_) {
            subscriber_lost_ = true;
            RCLCPP_WARN(this->get_logger(), "motion_commands has no matched ROS subscribers; input may be fresh but commands cannot be delivered");
        }
        const std::string key = s.source + ":" + s.io_kind + ":" + status;
        const bool healthy = status == "fresh" || status == "event_source_unverified" || status == "no_source";
        if (key != last_freshness_status_ || (!healthy && now >= next_freshness_log_ns_)) {
            const std::string summary = "remote freshness source=" + s.source + " device=" + s.device +
                " status=" + status + " io_kind=" + s.io_kind + " usb_age_ms=" + std::to_string(remote_controller::freshness_age_ms(s.io_ns, now)) +
                " valid_age_ms=" + std::to_string(remote_controller::freshness_age_ms(s.sample_ns, now)) +
                " last_stop=" + s.stop_status + " sample_hz=" + std::to_string(sample_hz) + " seq=" + std::to_string(s.samples) + " crc_errors=" + std::to_string(s.crc_errors);
            if (healthy) RCLCPP_INFO(this->get_logger(), "%s", summary.c_str());
            else RCLCPP_WARN(this->get_logger(), "%s", summary.c_str());
            last_freshness_status_ = key;
            next_freshness_log_ns_ = now + 5000000000LL;
        }
    }

    mutable std::mutex lock_;
    InputMapper mapper_;
    std::unique_ptr<remote_controller::InputDeviceManager> input_manager_;
    std::map<std::string, bool> system_mutex_locked_;
    bool has_last_published_payload_ = false;
    communication::msg::MotionCommands last_published_payload_;

    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::Publisher<communication::msg::MotionCommands>::SharedPtr com_pub_;

    void print_config_diagnostics(const RemoteConfig &config)
    {
        for (const auto &diagnostic : config.diagnostics) {
            if (diagnostic.severity == "warning") {
                RCLCPP_WARN(this->get_logger(), "%s", diagnostic.message.c_str());
            } else {
                RCLCPP_INFO(this->get_logger(), "%s", diagnostic.message.c_str());
            }
        }
    }

    void timer_callback()
    {
        const auto now = remote_controller::freshness_now_ns();
        if (last_timer_ns_ && now - last_timer_ns_ > 100000000LL) {
            ++timer_gaps_;
            if (now >= next_timer_gap_log_ns_) {
                RCLCPP_WARN(this->get_logger(), "remote publish loop gap: %.1f ms (gaps=%llu); check process scheduling/system load",
                    (now - last_timer_ns_) / 1e6, static_cast<unsigned long long>(timer_gaps_));
                next_timer_gap_log_ns_ = now + 5000000000LL;
            }
        }
        last_timer_ns_ = now;
        const bool must_publish_safe_command = input_manager_ && input_manager_->tick();
        auto message = communication::msg::MotionCommands();
        std::vector<std::string> outputs;
        bool publish_on_change = true;
        {
            const std::lock_guard<std::mutex> guard(lock_);
            outputs = mapper_.tick();
            mapper_.fill_message(message);
            publish_on_change = mapper_.config().publish_on_change;
        }
        dispatch_outputs(outputs);
        if (publish_on_change) {
            if (!must_publish_safe_command &&
                has_last_published_payload_ && message == last_published_payload_) {
                return;
            }

            last_published_payload_ = message;
            has_last_published_payload_ = true;
        }

        message.header.stamp = this->now();
        message.header.frame_id = "remote_controller";
        com_pub_->publish(message);
        ++published_count_;
        last_publish_ns_ = remote_controller::freshness_now_ns();
        if (must_publish_safe_command && input_manager_) {
            input_manager_->notify_safe_output_published();
        }
    }

    void dispatch_outputs(const std::vector<std::string> &outputs)
    {
        for (const auto &output : outputs) {
            if (remote_controller::starts_with(output, "system.")) {
                run_system_action(output.substr(std::string("system.").size()));
            } else if (!output.empty()) {
                RCLCPP_WARN(this->get_logger(), "unknown binding output: %s", output.c_str());
            }
        }
    }

    void run_system_action(const std::string &action)
    {
        std::vector<std::string> commands;
        bool reset_motion_after_system = false;
        {
            const std::lock_guard<std::mutex> guard(lock_);
            const std::string blocked_by = blocking_system_mutex_locked(action);
            if (!blocked_by.empty()) {
                RCLCPP_WARN(
                    this->get_logger(),
                    "system.%s ignored because mutex '%s' is already acquired",
                    action.c_str(),
                    blocked_by.c_str());
                return;
            }

            const auto &system_commands = mapper_.config().system_commands;
            const auto command_it = system_commands.find(action);
            if (command_it == system_commands.end()) {
                RCLCPP_WARN(this->get_logger(), "unknown system output: system.%s", action.c_str());
                return;
            }
            commands = command_it->second;
            reset_motion_after_system = mapper_.config().reset_motion_after_system.count(action) > 0;
        }

        run_commands(commands);

        {
            const std::lock_guard<std::mutex> guard(lock_);
            update_system_mutexes_locked(action);
            if (reset_motion_after_system) {
                mapper_.reset_motion();
            }
        }
    }

    std::string blocking_system_mutex_locked(const std::string &action) const
    {
        for (const auto &mutex : mapper_.config().system_mutexes) {
            if (mutex.acquire != action) {
                continue;
            }
            const auto lock_it = system_mutex_locked_.find(mutex.name);
            if (lock_it != system_mutex_locked_.end() && lock_it->second) {
                return mutex.name;
            }
        }
        return "";
    }

    void update_system_mutexes_locked(const std::string &action)
    {
        for (const auto &mutex : mapper_.config().system_mutexes) {
            if (mutex.release == action) {
                system_mutex_locked_[mutex.name] = false;
            }
            if (mutex.acquire == action) {
                system_mutex_locked_[mutex.name] = true;
            }
        }
    }

    void run_commands(const std::vector<std::string> &commands)
    {
        for (const auto &command : commands) {
            const int ret = std::system(command.c_str());
            (void)ret;
        }
    }
};

int main(int argc, const char *argv[])
{
    std::string driver_filter;
    std::string config_path;
    bool driver_debug_enabled = false;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--keyboard") {
            printf("in keyboard input mode\n");
            driver_filter = "keyboard";
        } else if (arg == "--driver" && i + 1 < argc) {
            driver_filter = argv[++i];
        } else if (arg == "--config" && i + 1 < argc) {
            config_path = argv[++i];
        } else if (arg == "--DEBUG" || arg == "--debug") {
            driver_debug_enabled = true;
        }
    }

    if (config_path.empty()) {
        fprintf(stderr, "remote_controller requires --config <yaml_path>\n");
        return 1;
    }

    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<COMPublisher>(
        config_path,
        driver_filter,
        driver_debug_enabled));
    rclcpp::shutdown();

    return 0;
}
