#include <rclcpp/rclcpp.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2/LinearMath/Quaternion.h>
#include <pcl/point_cloud.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>  
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/common/centroid.h>
#include <pcl/common/pca.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <Eigen/Eigen>
#include <memory>
#include <cmath>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

using namespace std::chrono_literals;

class RobotSimulator : public rclcpp::Node {
public:
    RobotSimulator() : Node("robot_simulator"), tf_broadcaster_(this) {
        // Declare parameters
        declare_parameters();

        // Initialize ROS interfaces
        init_ros_interfaces();

        // Load point cloud
        load_point_cloud();

        // Initialize pose
        init_pose();
        
        // 主控制循环
        control_timer_ = create_wall_timer(
            std::chrono::duration<double>(1.0 / control_rate_),
            std::bind(&RobotSimulator::control_cycle, this)
        );
    }

private:
    // Parameter declaration
    void declare_parameters() {
        // Basic parameters
        declare_parameter<std::string>("mesh_resource", "package://simulation_env/meshes/robot.dae");
        declare_parameter<double>("mesh_scale", 0.0005);
        declare_parameter<double>("mesh_offset_x", -0.15);
        declare_parameter<double>("mesh_offset_y", -0.16);
        declare_parameter<double>("max_linear_vel", 1.0);
        declare_parameter<double>("max_angular_vel", M_PI);
        declare_parameter<double>("control_rate", 50.0);

        // Point cloud parameters
        declare_parameter<std::string>("pcd_file_path", "");
        declare_parameter<double>("handle_range", 15.0);
        declare_parameter<double>("ground_range", 0.25);
        declare_parameter<double>("voxel_size", 0.05);

        // Initial pose parameters
        declare_parameter<double>("init_x", 0.0);
        declare_parameter<double>("init_y", 0.0);
        declare_parameter<double>("init_z", 0.0);

        // Plane fitting parameters
        declare_parameter<double>("max_tilt_angle", 30.0);
        declare_parameter<double>("inlier_threshold", 0.1);

        // Get parameter values
        get_parameters();
    }
    
    void get_parameters() {
        mesh_resource_ = get_parameter("mesh_resource").as_string();
        mesh_scale_ = get_parameter("mesh_scale").as_double();
        mesh_offset_x_ = get_parameter("mesh_offset_x").as_double();
        mesh_offset_y_ = get_parameter("mesh_offset_y").as_double();
        max_linear_vel_ = get_parameter("max_linear_vel").as_double();
        max_angular_vel_ = get_parameter("max_angular_vel").as_double();
        control_rate_ = get_parameter("control_rate").as_double();
        pcd_file_path_ = get_parameter("pcd_file_path").as_string();
        handle_range_ = get_parameter("handle_range").as_double();
        ground_range_ = get_parameter("ground_range").as_double();
        voxel_size_ = get_parameter("voxel_size").as_double();
        
        init_x_ = get_parameter("init_x").as_double();
        init_y_ = get_parameter("init_y").as_double();
        init_z_ = get_parameter("init_z").as_double();

        max_tilt_angle_ = get_parameter("max_tilt_angle").as_double() * M_PI / 180.0;
        inlier_threshold_ = get_parameter("inlier_threshold").as_double();
    }
    
    // Initialize ROS interfaces
    void init_ros_interfaces() {
        model_pub_ = create_publisher<visualization_msgs::msg::Marker>("/robot_model", 10);
        odom_pub_ = create_publisher<nav_msgs::msg::Odometry>("/odom", 10);
        local_cloud_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>("/local_cloud", 1);
        ground_cloud_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>("/ground_cloud", 1);
        plane_marker_pub_ = create_publisher<visualization_msgs::msg::Marker>("/ground_plane", 1);
        cmd_vel_sub_ = create_subscription<geometry_msgs::msg::Twist>(
            "/cmd_vel", 10, std::bind(&RobotSimulator::cmd_vel_callback, this, std::placeholders::_1));
    }
    
    // Load point cloud
    void load_point_cloud() {
        world_cloud_.reset(new pcl::PointCloud<pcl::PointXYZ>());
        local_cloud_.reset(new pcl::PointCloud<pcl::PointXYZ>());
        
        if (!pcd_file_path_.empty()) {
            if (pcl::io::loadPCDFile<pcl::PointXYZ>(pcd_file_path_, *world_cloud_) == -1) {
                RCLCPP_ERROR(get_logger(), "Failed to load PCD file: %s", pcd_file_path_.c_str());
                generate_test_cloud();
            } else {
                // Downsample the loaded point cloud
                pcl::PointCloud<pcl::PointXYZ>::Ptr downsampled_cloud(new pcl::PointCloud<pcl::PointXYZ>());
                pcl::VoxelGrid<pcl::PointXYZ> voxel_grid;
                voxel_grid.setInputCloud(world_cloud_);
                voxel_grid.setLeafSize(voxel_size_, voxel_size_, voxel_size_);
                voxel_grid.filter(*downsampled_cloud);
                *world_cloud_ = *downsampled_cloud;
                
                RCLCPP_INFO(get_logger(), "Downsampled cloud from %zu to %zu points", 
                        downsampled_cloud->size(), world_cloud_->size());
            }
        } else {
            generate_test_cloud();
        }
        
        world_cloud_->header.frame_id = "map";
        kd_tree_.setInputCloud(world_cloud_);
        RCLCPP_INFO(get_logger(), "Loaded %zu points", world_cloud_->size());
    }
    
    void generate_test_cloud() {
        // Generate test terrain
        for (float x = -50; x <= 50; x += 1.0) {
            for (float y = -50; y <= 50; y += 1.0) {
                // Add some terrain variation
                float z = 0.2 * sin(x * 0.2) * cos(y * 0.2);
                world_cloud_->push_back(pcl::PointXYZ(x, y, z));
            }
        }
        RCLCPP_WARN(get_logger(), "Generated test cloud with %zu points", world_cloud_->size());
    }
    
    // Initialize pose
    void init_pose() {
        current_pose_.position.x = init_x_;
        current_pose_.position.y = init_y_;
        current_pose_.position.z = init_z_;
        current_pose_.orientation = tf2::toMsg(tf2::Quaternion::getIdentity());
    }
    
    // Control loop
    void control_cycle() {
        // Extract local point cloud
        extract_local_cloud();

        // Update pose
        update_pose();

        // Publish all data
        publish_tf_and_odometry();
        publish_local_cloud();
        publish_ground_plane();
        publish_robot_model();
    }
    
    // Velocity callback
    void cmd_vel_callback(const geometry_msgs::msg::Twist::SharedPtr msg) {
        linear_vel_ = std::clamp(msg->linear.x, -max_linear_vel_, max_linear_vel_);
        angular_vel_ = std::clamp(msg->angular.z, -max_angular_vel_, max_angular_vel_);
    }
    
    // Publish TF and odometry
    void publish_tf_and_odometry() {
        // Publish TF
        geometry_msgs::msg::TransformStamped transform;
        transform.header.stamp = now();
        transform.header.frame_id = "map";
        transform.child_frame_id = "base_link";
        
        transform.transform.translation.x = current_pose_.position.x;
        transform.transform.translation.y = current_pose_.position.y;
        transform.transform.translation.z = current_pose_.position.z;
        transform.transform.rotation = current_pose_.orientation;
        
        tf_broadcaster_.sendTransform(transform);

        // Publish odometry
        auto odom_msg = nav_msgs::msg::Odometry();
        odom_msg.header.stamp = now();
        odom_msg.header.frame_id = "map";
        odom_msg.child_frame_id = "base_link";
        
        odom_msg.pose.pose = current_pose_;
        odom_msg.twist.twist.linear.x = linear_vel_;
        odom_msg.twist.twist.angular.z = angular_vel_;
        
        odom_pub_->publish(odom_msg);
    }
    
    // Extract local point cloud
    void extract_local_cloud() {
        pcl::PointXYZ search_point(
            current_pose_.position.x,
            current_pose_.position.y,
            current_pose_.position.z
        );
        
        std::vector<int> point_indices;
        std::vector<float> point_distances;
        
        local_cloud_.reset(new pcl::PointCloud<pcl::PointXYZ>());
        
        if (kd_tree_.radiusSearch(search_point, handle_range_, point_indices, point_distances) > 0) {
            for (const auto& idx : point_indices) {
                local_cloud_->points.push_back(world_cloud_->points[idx]);
            }
            local_cloud_->header = world_cloud_->header;
        }
    }
    
    // Publish local point cloud
    void publish_local_cloud() {
        if (!local_cloud_ || local_cloud_->empty()) return;
        
        sensor_msgs::msg::PointCloud2 cloud_msg;
        pcl::toROSMsg(*local_cloud_, cloud_msg);
        cloud_msg.header.stamp = now();
        cloud_msg.header.frame_id = "map";
        local_cloud_pub_->publish(cloud_msg);
    }
    
    // Update pose
    void update_pose() {
        const double dt = 1.0 / control_rate_;

        // 2D motion update
        update_2d_motion(dt);

        // Compute height and orientation from local point cloud
        if (local_cloud_ && !local_cloud_->empty()) {
            update_height_and_orientation();
        }
    }
    
    // 2D motion update
    void update_2d_motion(double dt) {
        if (std::abs(angular_vel_) > 1e-5) {
            const double radius = linear_vel_ / angular_vel_;
            current_pose_.position.x += radius * (
                sin(current_yaw_ + angular_vel_ * dt) - sin(current_yaw_)
            );
            current_pose_.position.y += radius * (
                cos(current_yaw_) - cos(current_yaw_ + angular_vel_ * dt)
            );
        } else {
            current_pose_.position.x += linear_vel_ * dt * cos(current_yaw_);
            current_pose_.position.y += linear_vel_ * dt * sin(current_yaw_);
        }
        
        current_yaw_ = fmod(current_yaw_ + angular_vel_ * dt, 2 * M_PI);
    }
    
    // Update height and orientation from local point cloud
    void update_height_and_orientation() {
        // Extract ground points
        pcl::PointCloud<pcl::PointXYZ>::Ptr ground_cloud(new pcl::PointCloud<pcl::PointXYZ>());
        extract_ground_points(ground_cloud);
        
        if (ground_cloud->size() < 10) {
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000, 
                               "Not enough ground points: %zu", ground_cloud->size());
            return;
        }
        
        // Use PCA to fit ground plane
        pcl::PCA<pcl::PointXYZ> pca;
        pca.setInputCloud(ground_cloud);

        Eigen::Vector3f eigen_values = pca.getEigenValues().cast<float>();
        Eigen::Matrix3f eigen_vectors = pca.getEigenVectors().cast<float>();

        // Normal vector is the eigenvector corresponding to the smallest eigenvalue
        Eigen::Vector3f normal = eigen_vectors.col(2);
        if (normal.z() < 0) normal = -normal;

        // Compute centroid
        Eigen::Vector4f centroid;
        pcl::compute3DCentroid(*ground_cloud, centroid);

        // Update pose
        update_pose_from_ground(normal, centroid);

        // Publish ground cloud for visualization
        publish_ground_cloud(ground_cloud);
    }
    
    // Extract ground points
    void extract_ground_points(pcl::PointCloud<pcl::PointXYZ>::Ptr& ground_cloud) {
        // Height-based statistical filtering
        std::vector<float> heights;
        for (const auto& point : local_cloud_->points) {
            double distance = sqrt(pow(point.x - current_pose_.position.x, 2) + 
                                 pow(point.y - current_pose_.position.y, 2));
            if (distance < ground_range_) {
                heights.push_back(point.z);
            }
        }
        
        if (heights.empty()) return;
        
        // Compute height statistics
        float mean_height = std::accumulate(heights.begin(), heights.end(), 0.0f) / heights.size();
        float height_std = 0.0f;
        for (float h : heights) height_std += pow(h - mean_height, 2);
        height_std = sqrt(height_std / heights.size());

        // Filter ground points
        for (const auto& point : local_cloud_->points) {
            double distance = sqrt(pow(point.x - current_pose_.position.x, 2) + 
                                 pow(point.y - current_pose_.position.y, 2));
            if (distance < ground_range_ && abs(point.z - mean_height) < 2.0 * height_std) {
                ground_cloud->points.push_back(point);
            }
        }
    }
    
    // Update pose from ground information
    void update_pose_from_ground(const Eigen::Vector3f& normal, const Eigen::Vector4f& centroid) {
        // Compute height
        double a = normal.x(), b = normal.y(), c = normal.z();
        double d = -normal.dot(Eigen::Vector3f(centroid[0], centroid[1], centroid[2]));
        current_pose_.position.z = (-a * current_pose_.position.x - b * current_pose_.position.y - d) / c;

        // Compute orientation from ground normal (world frame)
        double ground_roll = -atan2(normal.y(), normal.z());
        double ground_pitch = atan2(normal.x(), normal.z());
        tf2::Quaternion ground_orientation;
        ground_orientation.setRPY(ground_roll, ground_pitch, 0);

        // Create yaw-only quaternion
        tf2::Quaternion yaw_only;
        yaw_only.setRPY(0, 0, current_yaw_);

        // Combine: ground orientation first, then yaw rotation
        tf2::Quaternion final_orientation = ground_orientation * yaw_only;

        current_pose_.orientation = tf2::toMsg(final_orientation);

        // Save ground information
        ground_normal_ = normal;
        ground_centroid_ = centroid;
    }

    
    // Publish ground point cloud
    void publish_ground_cloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud) {
        sensor_msgs::msg::PointCloud2 cloud_msg;
        pcl::toROSMsg(*cloud, cloud_msg);
        cloud_msg.header.stamp = now();
        cloud_msg.header.frame_id = "map";
        ground_cloud_pub_->publish(cloud_msg);
    }
    
    // Publish ground plane visualization
    void publish_ground_plane() {
        visualization_msgs::msg::Marker marker;
        marker.header.frame_id = "map";
        marker.header.stamp = now();
        marker.ns = "ground_plane";
        marker.id = 0;
        marker.type = visualization_msgs::msg::Marker::CUBE;
        marker.action = visualization_msgs::msg::Marker::ADD;
        
        marker.pose.position.x = current_pose_.position.x;
        marker.pose.position.y = current_pose_.position.y;
        marker.pose.position.z = current_pose_.position.z;
        
        // Set orientation based on normal vector
        double roll = -atan2(ground_normal_.y(), ground_normal_.z());
        double pitch = atan2(ground_normal_.x(), ground_normal_.z());
        tf2::Quaternion q;
        q.setRPY(roll, pitch, 0);
        marker.pose.orientation = tf2::toMsg(q);
        
        marker.scale.x = 2.0;
        marker.scale.y = 2.0;
        marker.scale.z = 0.02;
        
        marker.color.r = 0.2f;
        marker.color.g = 0.8f;
        marker.color.b = 0.2f;
        marker.color.a = 0.6f;
        
        plane_marker_pub_->publish(marker);
    }
    
    // Publish robot model
    void publish_robot_model() {
        auto marker = visualization_msgs::msg::Marker();
        marker.header.frame_id = "map";
        marker.header.stamp = now();
        marker.ns = "robot";
        marker.id = 0;
        marker.type = visualization_msgs::msg::Marker::MESH_RESOURCE;
        marker.action = visualization_msgs::msg::Marker::ADD;

        marker.mesh_resource = mesh_resource_;
        marker.mesh_use_embedded_materials = true;

        tf2::Quaternion tf_quat;
        tf2::fromMsg(current_pose_.orientation, tf_quat);
        double roll, pitch, yaw;
        tf2::Matrix3x3(tf_quat).getRPY(roll, pitch, yaw);

        // Apply mesh offset
        double dx = mesh_offset_x_;
        double dy = mesh_offset_y_;
        marker.pose.position.x = current_pose_.position.x + dx * cos(yaw) - dy * sin(yaw);
        marker.pose.position.y = current_pose_.position.y + dx * sin(yaw) + dy * cos(yaw);
        marker.pose.position.z = current_pose_.position.z;

        // Apply rotation
        tf2::Quaternion q;
        q.setRPY(M_PI / 2, 0, M_PI / 2);
        marker.pose.orientation = tf2::toMsg(tf_quat * q);

        marker.scale.x = marker.scale.y = marker.scale.z = mesh_scale_;
        model_pub_->publish(marker);
    }

    // Parameters
    std::string mesh_resource_;
    double mesh_scale_;
    double mesh_offset_x_;
    double mesh_offset_y_;
    double max_linear_vel_;
    double max_angular_vel_;
    double control_rate_;
    std::string pcd_file_path_;
    double handle_range_;
    double ground_range_;
    double voxel_size_;
    double init_x_, init_y_, init_z_;
    double max_tilt_angle_;
    double inlier_threshold_;
    
    // Point cloud related
    pcl::PointCloud<pcl::PointXYZ>::Ptr world_cloud_;
    pcl::PointCloud<pcl::PointXYZ>::Ptr local_cloud_;
    pcl::KdTreeFLANN<pcl::PointXYZ> kd_tree_;

    // Robot state
    geometry_msgs::msg::Pose current_pose_;
    double current_yaw_ = 0.0;
    double linear_vel_ = 0.0;
    double angular_vel_ = 0.0;

    // Ground information
    Eigen::Vector3f ground_normal_ = Eigen::Vector3f::UnitZ();
    Eigen::Vector4f ground_centroid_ = Eigen::Vector4f::Zero();

    // ROS interfaces
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr model_pub_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr local_cloud_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr ground_cloud_pub_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr plane_marker_pub_;
    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_sub_;
    rclcpp::TimerBase::SharedPtr control_timer_;
    tf2_ros::TransformBroadcaster tf_broadcaster_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<RobotSimulator>());
    rclcpp::shutdown();
    return 0;
}