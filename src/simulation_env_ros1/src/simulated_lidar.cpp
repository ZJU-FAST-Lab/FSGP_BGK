// Simulated LiDAR node for ROS1 Noetic
// Generates simulated LiDAR scans from global point cloud with boundary checking

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <nav_msgs/Odometry.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/common/transforms.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <Eigen/Dense>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_eigen/tf2_eigen.h>
#include <geometry_msgs/TransformStamped.h>
#include <random>
#include <memory>
#include <vector>

class SimulatedLidar {
public:
    SimulatedLidar(ros::NodeHandle& nh, ros::NodeHandle& pnh) 
        : nh_(nh), pnh_(pnh), tf_buffer_(), tf_listener_(tf_buffer_) {
        
        // Parameter configuration
        pnh_.param<double>("horizontal_resolution", horizontal_resolution_, 0.05);
        pnh_.param<int>("num_laser_lines", num_laser_lines_, 64);
        pnh_.param<double>("max_range", MAX_RANGE, 5.0);
        pnh_.param<double>("min_range", MIN_RANGE, 0.1);
        pnh_.param<double>("noise_stddev", noise_stddev_, 0.02);
        pnh_.param<std::string>("robot_frame", robot_frame_, "base_link");
        pnh_.param<std::string>("world_frame", world_frame_, "map");
        pnh_.param<double>("publish_rate", publish_rate_, 10.0);
        pnh_.param<double>("tf_lookup_timeout", tf_lookup_timeout_, 0.1);
        
        // Map boundary for penetration fix
        pnh_.param<double>("map_min_x", map_min_x_, -100.0);
        pnh_.param<double>("map_max_x", map_max_x_, 100.0);
        pnh_.param<double>("map_min_y", map_min_y_, -100.0);
        pnh_.param<double>("map_max_y", map_max_y_, 100.0);
        pnh_.param<double>("map_min_z", map_min_z_, -10.0);
        pnh_.param<double>("map_max_z", map_max_z_, 50.0);
        pnh_.param<bool>("enable_boundary_check", enable_boundary_check_, true);
        
        // Initialize bin matrix
        int horizontal_bins = static_cast<int>(2 * M_PI / horizontal_resolution_);
        bin_matrix_ = std::vector<std::vector<double>>(horizontal_bins,
                     std::vector<double>(num_laser_lines_, std::numeric_limits<double>::max()));

        // Initialize random number generator
        random_engine_ = std::mt19937(std::random_device{}());
        noise_distribution_ = std::normal_distribution<double>(0.0, noise_stddev_);

        // Initialize publishers
        lidar_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("/simulated_lidar", 1);
        local_cloud_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("/simulated_lidar_local", 1);

        // Subscribe to global point cloud
        global_cloud_sub_ = nh_.subscribe("/local_cloud", 1, &SimulatedLidar::globalCloudCallback, this);

        // Timer
        timer_ = nh_.createTimer(ros::Duration(1.0 / publish_rate_), &SimulatedLidar::timerCallback, this);

        global_cloud_loaded_ = false;

        ROS_INFO("SimulatedLidar initialized: %d horizontal bins, %d laser lines, %.1f Hz", 
                 horizontal_bins, num_laser_lines_, publish_rate_);
        ROS_INFO("Using TF transform from %s to %s", world_frame_.c_str(), robot_frame_.c_str());
        
        if (enable_boundary_check_) {
            ROS_INFO("Boundary check enabled: X[%.1f, %.1f] Y[%.1f, %.1f] Z[%.1f, %.1f]",
                     map_min_x_, map_max_x_, map_min_y_, map_max_y_, map_min_z_, map_max_z_);
        }
    }

private:
    // Parameters
    double MAX_RANGE, MIN_RANGE;
    double horizontal_resolution_;
    int num_laser_lines_;
    double noise_stddev_;
    std::string robot_frame_, world_frame_;
    double publish_rate_, tf_lookup_timeout_;
    
    // Map boundary
    double map_min_x_, map_max_x_;
    double map_min_y_, map_max_y_;
    double map_min_z_, map_max_z_;
    bool enable_boundary_check_;

    ros::NodeHandle nh_, pnh_;
    ros::Publisher lidar_pub_, local_cloud_pub_;
    ros::Subscriber global_cloud_sub_;
    ros::Timer timer_;

    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_;

    pcl::PointCloud<pcl::PointXYZ>::Ptr global_cloud_;
    bool global_cloud_loaded_;
    std::vector<std::vector<double>> bin_matrix_;
    std::mt19937 random_engine_;
    std::normal_distribution<double> noise_distribution_;

    bool getCurrentTransform(Eigen::Affine3d& transform) {
        try {
            geometry_msgs::TransformStamped transform_stamped;
            transform_stamped = tf_buffer_.lookupTransform(
                world_frame_, robot_frame_, ros::Time(0), ros::Duration(tf_lookup_timeout_));
            transform = tf2::transformToEigen(transform_stamped.transform);
            return true;
        } catch (tf2::TransformException &ex) {
            ROS_WARN_THROTTLE(2.0, "TF lookup failed: %s", ex.what());
            return false;
        }
    }

    void globalCloudCallback(const sensor_msgs::PointCloud2::ConstPtr& msg) {
        if (global_cloud_loaded_) return;
        
        try {
            pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
            pcl::fromROSMsg(*msg, *cloud);
            
            if (cloud->empty()) {
                ROS_WARN("Received empty point cloud");
                return;
            }
            
            if (enable_boundary_check_) {
                updateMapBoundary(cloud);
            }
            
            global_cloud_ = cloud;
            global_cloud_loaded_ = true;
            
            ROS_INFO("Global cloud loaded with %zu points", cloud->size());
        } catch (const std::exception& e) {
            ROS_ERROR("Error processing global cloud: %s", e.what());
        }
    }

    void updateMapBoundary(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud) {
        if (cloud->empty()) return;

        map_min_x_ = map_min_y_ = map_min_z_ = std::numeric_limits<double>::max();
        map_max_x_ = map_max_y_ = map_max_z_ = std::numeric_limits<double>::lowest();

        for (const auto& point : cloud->points) {
            if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z)) continue;

            map_min_x_ = std::min(map_min_x_, (double)point.x);
            map_max_x_ = std::max(map_max_x_, (double)point.x);
            map_min_y_ = std::min(map_min_y_, (double)point.y);
            map_max_y_ = std::max(map_max_y_, (double)point.y);
            map_min_z_ = std::min(map_min_z_, (double)point.z);
            map_max_z_ = std::max(map_max_z_, (double)point.z);
        }

        double margin = 0.5;
        map_min_x_ -= margin; map_max_x_ += margin;
        map_min_y_ -= margin; map_max_y_ += margin;
        map_min_z_ -= margin; map_max_z_ += margin;

        ROS_INFO("Map boundary updated: X[%.2f, %.2f] Y[%.2f, %.2f] Z[%.2f, %.2f]",
                 map_min_x_, map_max_x_, map_min_y_, map_max_y_, map_min_z_, map_max_z_);
    }

    bool isPointInMapBoundary(double x, double y, double z) {
        if (!enable_boundary_check_) return true;
        return (x >= map_min_x_ && x <= map_max_x_ &&
                y >= map_min_y_ && y <= map_max_y_ &&
                z >= map_min_z_ && z <= map_max_z_);
    }

    bool isValidPoint(const pcl::PointXYZ& point) {
        return std::isfinite(point.x) && std::isfinite(point.y) && std::isfinite(point.z);
    }

    void resetBinMatrix() {
        static std::vector<double> default_row(num_laser_lines_, std::numeric_limits<double>::max());
        std::fill(bin_matrix_.begin(), bin_matrix_.end(), default_row);
    }

    void fillBinMatrix(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud) {
        for (size_t i = 0; i < cloud->size(); ++i) {
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

            h_bin = std::max(0, std::min(h_bin, static_cast<int>(bin_matrix_.size()-1)));
            v_bin = std::max(0, std::min(v_bin, static_cast<int>(bin_matrix_[0].size()-1)));

            if (range_with_noise < bin_matrix_[h_bin][v_bin]) {
                bin_matrix_[h_bin][v_bin] = range_with_noise;
            }
        }
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr generateSimulatedCloud(const Eigen::Affine3d& robot_to_world) {
        pcl::PointCloud<pcl::PointXYZ>::Ptr simulated_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        simulated_cloud->reserve(bin_matrix_.size() * bin_matrix_[0].size() / 4);

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
                        if (enable_boundary_check_) {
                            Eigen::Vector3d world_pt = robot_to_world * Eigen::Vector3d(point.x, point.y, point.z);
                            if (!isPointInMapBoundary(world_pt.x(), world_pt.y(), world_pt.z())) {
                                continue;
                            }
                        }
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

    void timerCallback(const ros::TimerEvent&) {
        if (!global_cloud_loaded_) {
            ROS_INFO_THROTTLE(2.0, "Waiting for global cloud data...");
            return;
        }

        Eigen::Affine3d current_transform;
        if (!getCurrentTransform(current_transform)) {
            ROS_WARN_THROTTLE(2.0, "Failed to get current TF transform");
            return;
        }

        try {
            resetBinMatrix();

            // Transform point cloud to robot frame
            pcl::PointCloud<pcl::PointXYZ>::Ptr robot_frame_cloud =
                transformCloud(global_cloud_, current_transform.inverse());

            if (robot_frame_cloud->empty()) {
                ROS_WARN_THROTTLE(2.0, "Transformed cloud is empty");
                return;
            }

            fillBinMatrix(robot_frame_cloud);

            // Generate simulated point cloud (pass transform for boundary check)
            pcl::PointCloud<pcl::PointXYZ>::Ptr simulated_cloud = generateSimulatedCloud(current_transform);

            if (!simulated_cloud->empty()) {
                // 发布局部点云（base_link坐标系）
                sensor_msgs::PointCloud2 local_msg;
                pcl::toROSMsg(*simulated_cloud, local_msg);
                local_msg.header.stamp = ros::Time::now();
                local_msg.header.frame_id = robot_frame_;
                local_cloud_pub_.publish(local_msg);

                // Publish world frame point cloud
                pcl::PointCloud<pcl::PointXYZ>::Ptr world_cloud =
                    transformCloud(simulated_cloud, current_transform);
                sensor_msgs::PointCloud2 world_msg;
                pcl::toROSMsg(*world_cloud, world_msg);
                world_msg.header.stamp = ros::Time::now();
                world_msg.header.frame_id = world_frame_;
                lidar_pub_.publish(world_msg);
            }

        } catch (const std::exception& e) {
            ROS_ERROR_THROTTLE(1.0, "Error in timer callback: %s", e.what());
        }
    }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "simulated_lidar");
    ros::NodeHandle nh;
    ros::NodeHandle pnh("~");

    SimulatedLidar lidar(nh, pnh);
    ros::spin();

    return 0;
}

