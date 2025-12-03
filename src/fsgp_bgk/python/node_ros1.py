#!/usr/bin/env python
'''
MIT License

Copyright (c) 2025 Senming Tan (senmingtan5@gmail.com)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

import rospy
from sensor_msgs.msg import PointCloud2, PointField
from nav_msgs.msg import Odometry
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import Header
import numpy as np
from f_sgp_bgk import TraversabilityAnalyzer  
from tf.transformations import euler_from_quaternion, quaternion_from_euler, quaternion_matrix
from scipy.ndimage import uniform_filter
import yaml
from scipy.interpolate import RegularGridInterpolator
from scipy.sparse import csr_matrix
import os

class FSGP_BGK_Node:
    def __init__(self):
        rospy.init_node('FSGP_BGK_Node')
        
        # Get config path relative to this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, "../config/params.yaml")
        
        try:
            with open(config_path, "r") as file:
                config = yaml.safe_load(file)
        except Exception as e:
            rospy.logerr(f"Failed to load config file: {e}")
            return

        self.cloud_topic = config["cloud_topic"]  
        self.odom_topic = config["odom_topic"]   
        self.global_link = config["global_link"]   
        self.max_height = config["max_cloud_height"]   
        self.occupancy_threshold = config["occupancy_threshold"]   
        self.decay_rate = config["decay_rate"]   
        self.smooth_kernel_size = config["smooth_kernel_size"]   
        self.map_update_rate = config["map_update_rate"]   
        self.obstacle_threshold = 1 - config["obstacle_threshold"]
        self.binarization_condition = config["binarization_condition"]
        self.publish_resolution = config["publish_resolution"]
        
        self.analyzer = TraversabilityAnalyzer(config_path=config_path)
        self.max_radius = (self.analyzer.x_length + self.analyzer.y_length) / 4  

        self.global_pose = None  
        self.latest_pcl = None  

        self.grid_resolution = self.analyzer.resolution  
        self.grid_size = (int(self.max_radius * 2 // self.grid_resolution), int(self.max_radius * 2 // self.grid_resolution))
        self.grid_half = int(self.max_radius // self.grid_resolution)
        self.global_grid = csr_matrix(self.grid_size, dtype=np.float32)  
        self.global_grid_ldd = csr_matrix(self.grid_size, dtype=np.float32)  
        self.log_odds_grid = np.full(self.grid_size, np.log(self.occupancy_threshold / (1 - self.occupancy_threshold)))  

        self.high_res_resolution = self.publish_resolution
        self.high_res_half = int(self.max_radius / self.high_res_resolution)
        self.high_res_shape = (self.high_res_half * 2, self.high_res_half * 2)
        self.high_res_x = (np.arange(self.high_res_shape[0]) - self.high_res_half) * self.high_res_resolution
        self.high_res_y = (np.arange(self.high_res_shape[1]) - self.high_res_half) * self.high_res_resolution
        self.high_res_xx, self.high_res_yy = np.meshgrid(self.high_res_x, self.high_res_y, indexing='ij')
        self.high_res_points = np.column_stack((self.high_res_xx.flatten(), self.high_res_yy.flatten()))

        # ROS1 Subscribers
        self.sph_pcl_sub = rospy.Subscriber(self.cloud_topic, PointCloud2, self.elevation_cb, queue_size=3)
        self.odom_sub = rospy.Subscriber(self.odom_topic, Odometry, self.odom_cb, queue_size=3)

        # ROS1 Publishers
        self.traversability_pcl_pub = rospy.Publisher("traversability_pcl", PointCloud2, queue_size=3)
        self.traversability_pcl_ldd_pub = rospy.Publisher("traversability_pcl_ldd", PointCloud2, queue_size=3)

        # ROS1 Timer
        self.map_timer = rospy.Timer(rospy.Duration(self.map_update_rate), self.map_thread)

        self.header = Header()
        self.header.frame_id = self.global_link
        self.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1)
        ]
        
        rospy.loginfo("FSGP_BGK_Node initialized successfully")

    def odom_cb(self, msg):
        if msg is None:
            return

        position = msg.pose.pose.position  
        orientation = msg.pose.pose.orientation  

        try:
            roll, pitch, yaw = euler_from_quaternion([orientation.x, orientation.y, orientation.z, orientation.w])
            self.global_pose = [position.x, position.y, position.z, roll, pitch, yaw]
        except Exception as ex:
            rospy.logwarn(f'err: {ex}')

    def elevation_cb(self, msg):
        if msg is not None:
            self.latest_pcl = msg

    def pointcloud2_to_xyz(self, msg):
        points = np.frombuffer(msg.data, dtype=np.uint8).reshape(-1, msg.point_step)
        xyz = np.zeros((points.shape[0], 3), dtype=np.float32)
        xyz[:, 0] = points[:, msg.fields[0].offset:msg.fields[0].offset + 4].view(np.float32).reshape(-1)
        xyz[:, 1] = points[:, msg.fields[1].offset:msg.fields[1].offset + 4].view(np.float32).reshape(-1)
        xyz[:, 2] = points[:, msg.fields[2].offset:msg.fields[2].offset + 4].view(np.float32).reshape(-1)

        radius_sq = xyz[:, 0]**2 + xyz[:, 1]**2
        mask = (xyz[:, 2] < self.max_height) & (radius_sq < self.max_radius**2)
        return xyz[mask]

    def observation_model(self, traversability):
        if self.binarization_condition:
            log_odds = np.zeros_like(traversability)
            log_odds[traversability > self.obstacle_threshold] = np.log(0.95 / (1 - 0.95))  
            log_odds[traversability <= self.obstacle_threshold] = np.log(0.05 / (1 - 0.05))  
            return log_odds
        else:
            traversability = np.clip(traversability, 0.001, 0.999)
            return np.log(traversability / (1 - traversability))

    def update_global_grid(self, global_smpld_pcl):
        pose_x, pose_y = self.global_pose[0], self.global_pose[1]
        grid_indices = ((global_smpld_pcl[:, :2] - np.array([pose_x, pose_y])) / self.grid_resolution).astype(int)
        valid_indices = (grid_indices[:, 0] + self.grid_half >= 0) & (grid_indices[:, 0] + self.grid_half < self.grid_size[0]) & \
                        (grid_indices[:, 1] + self.grid_half >= 0) & (grid_indices[:, 1] + self.grid_half < self.grid_size[1])
        grid_indices = grid_indices[valid_indices]
        traversability = global_smpld_pcl[valid_indices, 3]

        x_indices = grid_indices[:, 0] + self.grid_half
        y_indices = grid_indices[:, 1] + self.grid_half
        self.log_odds_grid[x_indices, y_indices] += self.observation_model(traversability)
        self.log_odds_grid = np.clip(self.log_odds_grid, -10, 10)
        self.global_grid_ldd[x_indices, y_indices] = 1 / (1 + np.exp(-self.log_odds_grid[x_indices, y_indices]))

        self.global_grid[x_indices, y_indices] = traversability

        self.global_grid = csr_matrix(uniform_filter(self.global_grid.toarray(), size=self.smooth_kernel_size))
        self.global_grid_ldd = csr_matrix(uniform_filter(self.global_grid_ldd.toarray(), size=self.smooth_kernel_size))

    def map_thread(self, event):
        if self.latest_pcl is None or self.global_pose is None:
            return

        local_points_np = self.pointcloud2_to_xyz(self.latest_pcl)

        self.analyzer.update_map(self.global_pose, local_points_np)
        traversabilitys = self.analyzer.traversability
        grid = self.analyzer.grid
        mean = self.analyzer.mean

        intensity = 1.0 - traversabilitys
        smpld_pcl = np.column_stack((grid[:, 0], grid[:, 1], mean, intensity))

        position = np.array(self.global_pose[:3])
        orientation = np.array(quaternion_from_euler(*self.global_pose[3:]))
        global_smpld_pcl = self.transform_smpl_pcl(smpld_pcl, position, orientation)
        self.update_global_grid(global_smpld_pcl)

        self.publish_global_grid()

    def publish_global_grid(self):
        if self.global_pose is None:
            rospy.logwarn("Global pose is not available.")
            return

        grid_shape = self.global_grid.shape
        low_res_x = (np.arange(grid_shape[0]) - self.grid_half) * self.grid_resolution + self.global_pose[0]
        low_res_y = (np.arange(grid_shape[1]) - self.grid_half) * self.grid_resolution + self.global_pose[1]

        try:
            global_grid_dense = self.global_grid.toarray()
            global_grid_ldd_dense = self.global_grid_ldd.toarray()

            interp_intensity = RegularGridInterpolator((low_res_x, low_res_y), global_grid_dense, method='linear', bounds_error=False, fill_value=1)
            interp_intensity_ldd = RegularGridInterpolator((low_res_x, low_res_y), global_grid_ldd_dense, method='linear', bounds_error=False, fill_value=1)

            high_res_points_global = self.high_res_points + self.global_pose[:2]

            high_res_intensity = interp_intensity(high_res_points_global)
            high_res_intensity_ldd = interp_intensity_ldd(high_res_points_global)
        except Exception as e:
            rospy.logerr(f"Interpolation failed: {e}")
            return

        z = np.full_like(self.high_res_xx.flatten(), self.global_pose[2]) + self.analyzer.base_height

        radius = np.hypot(self.high_res_xx.flatten(), self.high_res_yy.flatten())
        high_res_intensity[radius > self.max_radius - self.high_res_resolution * 2] = 1
        high_res_cloud_data = np.stack([self.high_res_xx.flatten() + self.global_pose[0],
                                    self.high_res_yy.flatten() + self.global_pose[1],
                                    z, high_res_intensity], axis=1)
        self.header.stamp = rospy.Time.now()
        self.traversability_pcl_pub.publish(pc2.create_cloud(self.header, self.fields, high_res_cloud_data))

        high_res_intensity_ldd[radius > self.max_radius - self.high_res_resolution * 2] = 1
        high_res_cloud_data_ldd = np.stack([self.high_res_xx.flatten() + self.global_pose[0],
                                            self.high_res_yy.flatten() + self.global_pose[1],
                                            z, high_res_intensity_ldd], axis=1)
        self.traversability_pcl_ldd_pub.publish(pc2.create_cloud(self.header, self.fields, high_res_cloud_data_ldd))

    def transform_smpl_pcl(self, smpl_pcl, position, orientation):
        points = smpl_pcl[:, :3]
        rotation_matrix = quaternion_matrix(orientation)[:3, :3]
        transformed_points = np.dot(points, rotation_matrix.T) + position
        transformed_smpl_pcl = np.column_stack((transformed_points, smpl_pcl[:, 3]))
        return transformed_smpl_pcl


def main():
    try:
        node = FSGP_BGK_Node()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    main()

