#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/common/transforms.h>
#include <Eigen/Dense>
#include <random>
#include <memory>
#include <vector>
#include <pcl/kdtree/kdtree_flann.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_eigen/tf2_eigen.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>

class SimulatedLidar : public rclcpp::Node {
public:
    SimulatedLidar() : Node("simulated_lidar"), 
                      tf_buffer_(this->get_clock()),
                      tf_listener_(tf_buffer_) {
        
        // 参数配置
        this->declare_parameter<double>("horizontal_resolution", 0.05);
        this->declare_parameter<int>("num_laser_lines", 64);
        this->declare_parameter<double>("max_range", 5.0);
        this->declare_parameter<double>("min_range", 0.1);
        this->declare_parameter<double>("noise_stddev", 0.02);
        this->declare_parameter<std::string>("robot_frame", "base_link");
        this->declare_parameter<std::string>("world_frame", "map");
        this->declare_parameter<double>("publish_rate", 10.0);
        this->declare_parameter<double>("tf_lookup_timeout", 0.1);
        
        horizontal_resolution_ = this->get_parameter("horizontal_resolution").as_double();
        num_laser_lines_ = this->get_parameter("num_laser_lines").as_int();
        MAX_RANGE = this->get_parameter("max_range").as_double();
        MIN_RANGE = this->get_parameter("min_range").as_double();
        noise_stddev_ = this->get_parameter("noise_stddev").as_double();
        robot_frame_ = this->get_parameter("robot_frame").as_string();
        world_frame_ = this->get_parameter("world_frame").as_string();
        double publish_rate = this->get_parameter("publish_rate").as_double();
        tf_lookup_timeout_ = this->get_parameter("tf_lookup_timeout").as_double();
        
        // 初始化bin矩阵
        int horizontal_bins = static_cast<int>(2 * M_PI / horizontal_resolution_);
        bin_matrix_ = std::vector<std::vector<double>>(horizontal_bins, 
                     std::vector<double>(num_laser_lines_, std::numeric_limits<double>::max()));

        // 初始化随机数生成器
        random_engine_ = std::mt19937(std::random_device{}());
        noise_distribution_ = std::normal_distribution<double>(0.0, noise_stddev_);

        // 使用与RViz2兼容的QoS设置
        auto pointcloud_qos = rclcpp::QoS(rclcpp::KeepLast(5));
        pointcloud_qos.best_effort();
        pointcloud_qos.durability_volatile();

        // 初始化发布器
        lidar_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
            "/simulated_lidar", 1);
        local_cloud_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
            "/simulated_lidar_local", 1);
        
        // 订阅全局点云（只订阅一次）
        auto global_cloud_qos = rclcpp::QoS(rclcpp::KeepLast(1));
        global_cloud_qos.transient_local();
        global_cloud_qos.reliable();
        
        global_cloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/local_cloud", 1,
            std::bind(&SimulatedLidar::globalCloudCallback, this, std::placeholders::_1));
        
        // 使用可配置的发布频率
        int timer_interval_ms = static_cast<int>(1000.0 / publish_rate);
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(timer_interval_ms),
            std::bind(&SimulatedLidar::timerCallback, this));

        // 状态变量
        global_cloud_loaded_ = false;
        last_transform_valid_ = false;

        RCLCPP_INFO(this->get_logger(), 
                   "SimulatedLidar initialized: %d horizontal bins, %d laser lines, %.1f Hz publish rate", 
                   horizontal_bins, num_laser_lines_, publish_rate);
        RCLCPP_INFO(this->get_logger(), "Using TF transform from %s to %s", 
                   world_frame_.c_str(), robot_frame_.c_str());
    }

private:
    // 参数
    double MAX_RANGE;
    double MIN_RANGE;
    double horizontal_resolution_;
    int num_laser_lines_;
    double noise_stddev_;
    std::string robot_frame_;
    std::string world_frame_;
    double tf_lookup_timeout_;
    
    // ROS接口
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr lidar_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr local_cloud_pub_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr global_cloud_sub_;
    rclcpp::TimerBase::SharedPtr timer_;
    
    // TF2
    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_;
    
    // 数据存储
    pcl::PointCloud<pcl::PointXYZ>::Ptr global_cloud_;
    bool global_cloud_loaded_;
    bool last_transform_valid_;
    Eigen::Affine3d last_transform_;
    
    std::vector<std::vector<double>> bin_matrix_;
    
    // 随机数生成器
    std::mt19937 random_engine_;
    std::normal_distribution<double> noise_distribution_;

    bool getCurrentTransform(Eigen::Affine3d& transform) {
        try {
            // 使用最新的TF变换，避免时间同步问题
            geometry_msgs::msg::TransformStamped transform_stamped;
            
            // 尝试获取最新的变换
            transform_stamped = tf_buffer_.lookupTransform(
                world_frame_, robot_frame_, tf2::TimePointZero,
                tf2::durationFromSec(tf_lookup_timeout_));
            
            transform = tf2::transformToEigen(transform_stamped.transform);
            return true;
            
        } catch (tf2::TransformException &ex) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                               "TF lookup failed: %s", ex.what());
            return false;
        }
    }

    void globalCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        // 只处理第一次接收到的点云
        if (global_cloud_loaded_) {
            return;
        }
        
        try {
            pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
            pcl::fromROSMsg(*msg, *cloud);
            
            if (cloud->empty()) {
                RCLCPP_WARN(this->get_logger(), "Received empty point cloud");
                return;
            }
            
            global_cloud_ = cloud;
            global_cloud_loaded_ = true;
            
            RCLCPP_INFO(this->get_logger(), 
                       "Global cloud loaded with %zu points, will not update again", 
                       cloud->size());
            
            // 成功加载后，取消订阅以节省资源
            global_cloud_sub_.reset();
            
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Error processing global cloud: %s", e.what());
        }
    }

    bool isValidPoint(const pcl::PointXYZ& point) {
        return std::isfinite(point.x) && std::isfinite(point.y) && std::isfinite(point.z);
    }

    void resetBinMatrix() {
        static std::vector<double> default_row(num_laser_lines_, std::numeric_limits<double>::max());
        std::fill(bin_matrix_.begin(), bin_matrix_.end(), default_row);
    }

    void fillBinMatrix(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud) {
        const size_t cloud_size = cloud->size();
        
        for (size_t i = 0; i < cloud_size; ++i) {
            const auto& point = cloud->points[i];
            if (!isValidPoint(point)) continue;

            const double x = point.x;
            const double y = point.y;
            const double z = point.z;
            
            const double range_sq = x*x + y*y + z*z;
            if (range_sq > MAX_RANGE*MAX_RANGE || range_sq < MIN_RANGE*MIN_RANGE) continue;

            const double range = std::sqrt(range_sq);
            const double range_with_noise = range + noise_distribution_(random_engine_);
            
            if (range_with_noise > MAX_RANGE || range_with_noise < MIN_RANGE) continue;

            const double horizontal_angle = std::atan2(y, x);
            const double vertical_angle = std::atan2(z, std::hypot(x, y));

            int h_bin = static_cast<int>((horizontal_angle + M_PI) / horizontal_resolution_);
            int v_bin = static_cast<int>((vertical_angle + M_PI/2) / (M_PI/(num_laser_lines_-1)));
            
            h_bin = std::clamp(h_bin, 0, static_cast<int>(bin_matrix_.size()-1));
            v_bin = std::clamp(v_bin, 0, static_cast<int>(bin_matrix_[0].size()-1));
            
            if (range_with_noise < bin_matrix_[h_bin][v_bin]) {
                bin_matrix_[h_bin][v_bin] = range_with_noise;
            }
        }
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr generateSimulatedCloud() {
        pcl::PointCloud<pcl::PointXYZ>::Ptr simulated_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        const size_t expected_size = bin_matrix_.size() * bin_matrix_[0].size() / 4;
        simulated_cloud->reserve(expected_size);

        for (size_t h = 0; h < bin_matrix_.size(); ++h) {
            for (size_t v = 0; v < bin_matrix_[0].size(); ++v) {
                if (bin_matrix_[h][v] < std::numeric_limits<double>::max()) {
                    const double h_angle = h * horizontal_resolution_ - M_PI;
                    const double v_angle = v * (M_PI/(num_laser_lines_-1)) - M_PI/2;
                    const double range = bin_matrix_[h][v];

                    pcl::PointXYZ point;
                    point.x = range * std::cos(v_angle) * std::cos(h_angle);
                    point.y = range * std::cos(v_angle) * std::sin(h_angle);
                    point.z = range * std::sin(v_angle);
                    
                    if (isValidPoint(point)) {
                        simulated_cloud->push_back(point);
                    }
                }
            }
        }
        
        simulated_cloud->width = simulated_cloud->size();
        simulated_cloud->height = 1;
        simulated_cloud->is_dense = true;
        
        return simulated_cloud;
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr transformCloud(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, 
        const Eigen::Affine3d& transform) {
        
        pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        transformed_cloud->reserve(cloud->size());
        
        for (const auto& point : cloud->points) {
            Eigen::Vector3d transformed = transform * Eigen::Vector3d(point.x, point.y, point.z);
            pcl::PointXYZ new_point;
            new_point.x = transformed.x();
            new_point.y = transformed.y();
            new_point.z = transformed.z();
            
            if (isValidPoint(new_point)) {
                transformed_cloud->push_back(new_point);
            }
        }
        
        transformed_cloud->width = transformed_cloud->size();
        transformed_cloud->height = 1;
        transformed_cloud->is_dense = true;
        
        return transformed_cloud;
    }

    void publishPointCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, 
                          const std::string& frame_id,
                          rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr publisher,
                          const std::string& cloud_name) {
        if (!cloud->empty()) {
            sensor_msgs::msg::PointCloud2 output;
            pcl::toROSMsg(*cloud, output);
            output.header.stamp = this->now();
            output.header.frame_id = frame_id;
            publisher->publish(output);
            
            RCLCPP_DEBUG_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                                "Published %s: %zu points", cloud_name.c_str(), cloud->size());
        }
    }

    void timerCallback() {
        if (!global_cloud_loaded_) {
            RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                               "Waiting for global cloud data...");
            return;
        }

        // 获取当前TF变换
        Eigen::Affine3d current_transform;
        if (!getCurrentTransform(current_transform)) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                               "Failed to get current TF transform");
            return;
        }

        try {
            auto start_time = std::chrono::high_resolution_clock::now();
            
            // 1. 重置bin矩阵
            resetBinMatrix();
            
            // 2. 转换点云到机器人坐标系
            pcl::PointCloud<pcl::PointXYZ>::Ptr robot_frame_cloud = 
                transformCloud(global_cloud_, current_transform.inverse());

            if (robot_frame_cloud->empty()) {
                RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                                   "Transformed cloud is empty");
                return;
            }

            // 3. 填充bin矩阵
            fillBinMatrix(robot_frame_cloud);

            // 4. 生成模拟点云
            pcl::PointCloud<pcl::PointXYZ>::Ptr simulated_cloud = generateSimulatedCloud();

            // 5. 发布点云
            if (!simulated_cloud->empty()) {
                // 发布局部点云（base_link坐标系）- 直接使用生成的模拟点云
                publishPointCloud(simulated_cloud, robot_frame_, local_cloud_pub_, "local frame lidar");
                
                // 发布世界坐标系点云 - 转换回世界坐标系
                pcl::PointCloud<pcl::PointXYZ>::Ptr world_cloud = 
                    transformCloud(simulated_cloud, current_transform);
                publishPointCloud(world_cloud, world_frame_, lidar_pub_, "world frame lidar");
            }

            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
            
            RCLCPP_DEBUG_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                                "Processing time: %ld ms, Points: %zu", 
                                duration.count(), simulated_cloud->size());
            
            // 更新最后有效的变换
            last_transform_ = current_transform;
            last_transform_valid_ = true;
            
        } catch (const std::exception& e) {
            RCLCPP_ERROR_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                                "Error in timer callback: %s", e.what());
        }
    }
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    try {
        auto node = std::make_shared<SimulatedLidar>();
        rclcpp::spin(node);
    } catch (const std::exception& e) {
        std::cerr << "Exception in main: " << e.what() << std::endl;
        return 1;
    }
    rclcpp::shutdown();
    return 0;
}