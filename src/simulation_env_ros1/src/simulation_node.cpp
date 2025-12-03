// Robot simulation node for ROS1 Noetic
// Simulates robot motion on 3D terrain with height adaptation

#include <ros/ros.h>
#include <visualization_msgs/Marker.h>
#include <geometry_msgs/Twist.h>
#include <geometry_msgs/Pose.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <pcl/point_cloud.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/common/centroid.h>
#include <pcl/common/pca.h>
#include <pcl_conversions/pcl_conversions.h>
#include <Eigen/Eigen>
#include <memory>
#include <cmath>
#include <numeric>

class RobotSimulator {
public:
    RobotSimulator(ros::NodeHandle& nh, ros::NodeHandle& pnh) : nh_(nh), pnh_(pnh) {
        // 获取参数
        load_parameters();
        
        // 初始化ROS接口
        init_ros_interfaces();
        
        // 加载点云
        load_point_cloud();
        
        // 初始化位姿
        init_pose();
        
        // 主控制循环定时器
        control_timer_ = nh_.createTimer(
            ros::Duration(1.0 / control_rate_),
            &RobotSimulator::control_cycle, this
        );
        
        ROS_INFO("RobotSimulator initialized successfully");
    }

private:
    void load_parameters() {
        // 基础参数
        pnh_.param<std::string>("mesh_resource", mesh_resource_, 
                                "package://simulation_env_ros1/meshes/robot.dae");
        pnh_.param<double>("mesh_scale", mesh_scale_, 0.0005);
        pnh_.param<double>("mesh_offset_x", mesh_offset_x_, -0.15);
        pnh_.param<double>("mesh_offset_y", mesh_offset_y_, -0.16);
        pnh_.param<double>("max_linear_vel", max_linear_vel_, 1.0);
        pnh_.param<double>("max_angular_vel", max_angular_vel_, M_PI);
        pnh_.param<double>("control_rate", control_rate_, 50.0);
        
        // 点云参数
        pnh_.param<std::string>("pcd_file_path", pcd_file_path_, "");
        pnh_.param<double>("handle_range", handle_range_, 15.0);
        pnh_.param<double>("ground_range", ground_range_, 0.25);
        pnh_.param<double>("voxel_size", voxel_size_, 0.05);
        
        // 初始化参数
        pnh_.param<double>("init_x", init_x_, 0.0);
        pnh_.param<double>("init_y", init_y_, 0.0);
        pnh_.param<double>("init_z", init_z_, 0.0);
        
        // 平面拟合参数
        pnh_.param<double>("max_tilt_angle", max_tilt_angle_, 30.0);
        max_tilt_angle_ = max_tilt_angle_ * M_PI / 180.0;
        pnh_.param<double>("inlier_threshold", inlier_threshold_, 0.1);
    }
    
    void init_ros_interfaces() {
        model_pub_ = nh_.advertise<visualization_msgs::Marker>("/robot_model", 10);
        odom_pub_ = nh_.advertise<nav_msgs::Odometry>("/odom", 10);
        local_cloud_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("/local_cloud", 1);
        ground_cloud_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("/ground_cloud", 1);
        plane_marker_pub_ = nh_.advertise<visualization_msgs::Marker>("/ground_plane", 1);
        cmd_vel_sub_ = nh_.subscribe("/cmd_vel", 10, &RobotSimulator::cmd_vel_callback, this);
    }
    
    void load_point_cloud() {
        world_cloud_.reset(new pcl::PointCloud<pcl::PointXYZ>());
        local_cloud_.reset(new pcl::PointCloud<pcl::PointXYZ>());
        
        if (!pcd_file_path_.empty()) {
            if (pcl::io::loadPCDFile<pcl::PointXYZ>(pcd_file_path_, *world_cloud_) == -1) {
                ROS_ERROR("Failed to load PCD file: %s", pcd_file_path_.c_str());
                generate_test_cloud();
            } else {
                // 降采样
                pcl::PointCloud<pcl::PointXYZ>::Ptr downsampled(new pcl::PointCloud<pcl::PointXYZ>());
                pcl::VoxelGrid<pcl::PointXYZ> voxel_grid;
                voxel_grid.setInputCloud(world_cloud_);
                voxel_grid.setLeafSize(voxel_size_, voxel_size_, voxel_size_);
                voxel_grid.filter(*downsampled);
                *world_cloud_ = *downsampled;
                ROS_INFO("Loaded and downsampled to %zu points", world_cloud_->size());
            }
        } else {
            generate_test_cloud();
        }
        
        world_cloud_->header.frame_id = "map";
        kd_tree_.setInputCloud(world_cloud_);
        ROS_INFO("Point cloud ready with %zu points", world_cloud_->size());
    }
    
    void generate_test_cloud() {
        for (float x = -50; x <= 50; x += 1.0) {
            for (float y = -50; y <= 50; y += 1.0) {
                float z = 0.2 * sin(x * 0.2) * cos(y * 0.2);
                world_cloud_->push_back(pcl::PointXYZ(x, y, z));
            }
        }
        ROS_WARN("Generated test cloud with %zu points", world_cloud_->size());
    }
    
    void init_pose() {
        current_pose_.position.x = init_x_;
        current_pose_.position.y = init_y_;
        current_pose_.position.z = init_z_;
        tf2::Quaternion q;
        q.setRPY(0, 0, 0);
        current_pose_.orientation = tf2::toMsg(q);
    }
    
    void control_cycle(const ros::TimerEvent&) {
        extract_local_cloud();
        update_pose();
        publish_tf_and_odometry();
        publish_local_cloud();
        publish_ground_plane();
        publish_robot_model();
    }
    
    void cmd_vel_callback(const geometry_msgs::Twist::ConstPtr& msg) {
        linear_vel_ = std::max(-max_linear_vel_, std::min(msg->linear.x, max_linear_vel_));
        angular_vel_ = std::max(-max_angular_vel_, std::min(msg->angular.z, max_angular_vel_));
    }

    void publish_tf_and_odometry() {
        geometry_msgs::TransformStamped transform;
        transform.header.stamp = ros::Time::now();
        transform.header.frame_id = "map";
        transform.child_frame_id = "base_link";

        transform.transform.translation.x = current_pose_.position.x;
        transform.transform.translation.y = current_pose_.position.y;
        transform.transform.translation.z = current_pose_.position.z;
        transform.transform.rotation = current_pose_.orientation;

        tf_broadcaster_.sendTransform(transform);

        nav_msgs::Odometry odom_msg;
        odom_msg.header.stamp = ros::Time::now();
        odom_msg.header.frame_id = "map";
        odom_msg.child_frame_id = "base_link";
        odom_msg.pose.pose = current_pose_;
        odom_msg.twist.twist.linear.x = linear_vel_;
        odom_msg.twist.twist.angular.z = angular_vel_;
        odom_pub_.publish(odom_msg);
    }

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

    void publish_local_cloud() {
        if (!local_cloud_ || local_cloud_->empty()) return;

        sensor_msgs::PointCloud2 cloud_msg;
        pcl::toROSMsg(*local_cloud_, cloud_msg);
        cloud_msg.header.stamp = ros::Time::now();
        cloud_msg.header.frame_id = "map";
        local_cloud_pub_.publish(cloud_msg);
    }

    void update_pose() {
        const double dt = 1.0 / control_rate_;
        update_2d_motion(dt);

        if (local_cloud_ && !local_cloud_->empty()) {
            update_height_and_orientation();
        }
    }

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

    void update_height_and_orientation() {
        pcl::PointCloud<pcl::PointXYZ>::Ptr ground_cloud(new pcl::PointCloud<pcl::PointXYZ>());
        extract_ground_points(ground_cloud);

        if (ground_cloud->size() < 3) {
            ROS_WARN_THROTTLE(1.0, "Not enough ground points: %zu", ground_cloud->size());
            return;
        }

        pcl::PCA<pcl::PointXYZ> pca;
        pca.setInputCloud(ground_cloud);

        Eigen::Matrix3f eigen_vectors = pca.getEigenVectors().cast<float>();
        Eigen::Vector3f normal = eigen_vectors.col(2);
        if (normal.z() < 0) normal = -normal;

        Eigen::Vector4f centroid;
        pcl::compute3DCentroid(*ground_cloud, centroid);

        update_pose_from_ground(normal, centroid);
        publish_ground_cloud(ground_cloud);
    }

    void extract_ground_points(pcl::PointCloud<pcl::PointXYZ>::Ptr& ground_cloud) {
        std::vector<float> heights;
        for (const auto& point : local_cloud_->points) {
            double distance = sqrt(pow(point.x - current_pose_.position.x, 2) +
                                 pow(point.y - current_pose_.position.y, 2));
            if (distance < ground_range_) {
                heights.push_back(point.z);
            }
        }

        if (heights.empty()) return;

        float mean_height = std::accumulate(heights.begin(), heights.end(), 0.0f) / heights.size();
        float height_std = 0.0f;
        for (float h : heights) height_std += pow(h - mean_height, 2);
        height_std = sqrt(height_std / heights.size());

        for (const auto& point : local_cloud_->points) {
            double distance = sqrt(pow(point.x - current_pose_.position.x, 2) +
                                 pow(point.y - current_pose_.position.y, 2));
            if (distance < ground_range_ && std::abs(point.z - mean_height) < 2.0 * height_std) {
                ground_cloud->points.push_back(point);
            }
        }
    }

    void update_pose_from_ground(const Eigen::Vector3f& normal, const Eigen::Vector4f& centroid) {
        double a = normal.x(), b = normal.y(), c = normal.z();
        double d = -normal.dot(Eigen::Vector3f(centroid[0], centroid[1], centroid[2]));
        current_pose_.position.z = (-a * current_pose_.position.x - b * current_pose_.position.y - d) / c;

        double ground_roll = -atan2(normal.y(), normal.z());
        double ground_pitch = atan2(normal.x(), normal.z());
        tf2::Quaternion ground_orientation;
        ground_orientation.setRPY(ground_roll, ground_pitch, 0);

        tf2::Quaternion yaw_only;
        yaw_only.setRPY(0, 0, current_yaw_);

        tf2::Quaternion final_orientation = ground_orientation * yaw_only;
        current_pose_.orientation = tf2::toMsg(final_orientation);

        ground_normal_ = normal;
        ground_centroid_ = centroid;
    }

    void publish_ground_cloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud) {
        sensor_msgs::PointCloud2 cloud_msg;
        pcl::toROSMsg(*cloud, cloud_msg);
        cloud_msg.header.stamp = ros::Time::now();
        cloud_msg.header.frame_id = "map";
        ground_cloud_pub_.publish(cloud_msg);
    }

    void publish_ground_plane() {
        visualization_msgs::Marker marker;
        marker.header.frame_id = "map";
        marker.header.stamp = ros::Time::now();
        marker.ns = "ground_plane";
        marker.id = 0;
        marker.type = visualization_msgs::Marker::CUBE;
        marker.action = visualization_msgs::Marker::ADD;

        marker.pose.position.x = current_pose_.position.x;
        marker.pose.position.y = current_pose_.position.y;
        marker.pose.position.z = current_pose_.position.z;

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

        plane_marker_pub_.publish(marker);
    }

    void publish_robot_model() {
        visualization_msgs::Marker marker;
        marker.header.frame_id = "map";
        marker.header.stamp = ros::Time::now();
        marker.ns = "robot";
        marker.id = 0;
        marker.type = visualization_msgs::Marker::MESH_RESOURCE;
        marker.action = visualization_msgs::Marker::ADD;

        marker.mesh_resource = mesh_resource_;
        marker.mesh_use_embedded_materials = true;

        tf2::Quaternion tf_quat;
        tf2::fromMsg(current_pose_.orientation, tf_quat);
        double roll, pitch, yaw;
        tf2::Matrix3x3(tf_quat).getRPY(roll, pitch, yaw);

        double dx = mesh_offset_x_;
        double dy = mesh_offset_y_;
        marker.pose.position.x = current_pose_.position.x + dx * cos(yaw) - dy * sin(yaw);
        marker.pose.position.y = current_pose_.position.y + dx * sin(yaw) + dy * cos(yaw);
        marker.pose.position.z = current_pose_.position.z;

        tf2::Quaternion q;
        q.setRPY(M_PI / 2, 0, M_PI / 2);
        tf2::Quaternion result = tf_quat * q;
        marker.pose.orientation = tf2::toMsg(result);

        marker.scale.x = marker.scale.y = marker.scale.z = mesh_scale_;
        model_pub_.publish(marker);
    }

    // 成员变量
    ros::NodeHandle nh_, pnh_;

    // 参数
    std::string mesh_resource_, pcd_file_path_;
    double mesh_scale_, mesh_offset_x_, mesh_offset_y_;
    double max_linear_vel_, max_angular_vel_, control_rate_;
    double handle_range_, ground_range_, voxel_size_;
    double init_x_, init_y_, init_z_;
    double max_tilt_angle_, inlier_threshold_;

    // 点云相关
    pcl::PointCloud<pcl::PointXYZ>::Ptr world_cloud_;
    pcl::PointCloud<pcl::PointXYZ>::Ptr local_cloud_;
    pcl::KdTreeFLANN<pcl::PointXYZ> kd_tree_;

    // 机器人状态
    geometry_msgs::Pose current_pose_;
    double current_yaw_ = 0.0;
    double linear_vel_ = 0.0;
    double angular_vel_ = 0.0;

    // 地面信息
    Eigen::Vector3f ground_normal_ = Eigen::Vector3f::UnitZ();
    Eigen::Vector4f ground_centroid_ = Eigen::Vector4f::Zero();

    // ROS接口
    ros::Publisher model_pub_, odom_pub_, local_cloud_pub_, ground_cloud_pub_, plane_marker_pub_;
    ros::Subscriber cmd_vel_sub_;
    ros::Timer control_timer_;
    tf2_ros::TransformBroadcaster tf_broadcaster_;
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "simulation_node");
    ros::NodeHandle nh;
    ros::NodeHandle pnh("~");

    RobotSimulator simulator(nh, pnh);
    ros::spin();

    return 0;
}

