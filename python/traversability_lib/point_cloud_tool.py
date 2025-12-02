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

import numpy as np
import open3d as o3d
import numpy as np

def pointcloud2_to_xyz(msg, z_threshold=2.0):
    field_names = [field.name for field in msg.fields]
    field_offsets = [field.offset for field in msg.fields]

    # Find the indices of the x, y, z fields
    x_idx = field_names.index("x") if "x" in field_names else None
    y_idx = field_names.index("y") if "y" in field_names else None
    z_idx = field_names.index("z") if "z" in field_names else None

    if x_idx is None or y_idx is None or z_idx is None:
        raise ValueError("PointCloud2 message must contain x, y, and z fields.")

    dtype = np.dtype([(name, np.float32) for name in ["x", "y", "z"]])
    point_step = msg.point_step
    num_points = len(msg.data) // point_step

    if len(msg.data) % point_step != 0:
        raise ValueError("PointCloud2 data size is not a multiple of point_step.")

    points = np.frombuffer(msg.data, dtype=np.uint8).reshape(-1, point_step)
    xyz = np.zeros((num_points, 3), dtype=np.float32)

    xyz[:, 0] = points[:, field_offsets[x_idx]:field_offsets[x_idx] + 4].view(np.float32).reshape(-1)
    xyz[:, 1] = points[:, field_offsets[y_idx]:field_offsets[y_idx] + 4].view(np.float32).reshape(-1)
    xyz[:, 2] = points[:, field_offsets[z_idx]:field_offsets[z_idx] + 4].view(np.float32).reshape(-1)

    mask = xyz[:, 2] < z_threshold
    filtered_xyz = xyz[mask]

    return filtered_xyz

def point_cloud2_to_array(msg, z_threshold):
    field_names = [field.name for field in msg.fields]
    intensity_idx = field_names.index("intensity") if "intensity" in field_names else None

    pc_data = np.frombuffer(msg.data, dtype=np.uint8).reshape(-1, msg.point_step)
    xyz = np.zeros((pc_data.shape[0], 3), dtype=np.float32)

    xyz[:, 0] = pc_data[:, msg.fields[field_names.index("x")].offset:msg.fields[field_names.index("x")].offset + 4].view(np.float32).reshape(-1)
    xyz[:, 1] = pc_data[:, msg.fields[field_names.index("y")].offset:msg.fields[field_names.index("y")].offset + 4].view(np.float32).reshape(-1)
    xyz[:, 2] = pc_data[:, msg.fields[field_names.index("z")].offset:msg.fields[field_names.index("z")].offset + 4].view(np.float32).reshape(-1)
    
    mask = xyz[:, 2] < z_threshold
    filtered_xyz = xyz[mask]

    if intensity_idx is not None:
        intensity = pc_data[mask, msg.fields[intensity_idx].offset:msg.fields[intensity_idx].offset + 4].view(np.float32).reshape(-1, 1)
    else:
        intensity = np.ones((filtered_xyz.shape[0], 1), dtype=np.float32)

    filtered_data = np.hstack((filtered_xyz, intensity))

    return filtered_data

def pointcloud2_to_array(cloud_msg,z_threshold):
    arr= point_cloud2_to_array(cloud_msg,z_threshold)   
    return arr

def voxel_downsample(pcl_arr, voxel_size):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcl_arr[:, :3])  

    downsampled_pcd = pcd.voxel_down_sample(voxel_size)
    downsampled_positions = np.asarray(downsampled_pcd.points)  
    
    if pcl_arr.shape[1] == 4:  
        intensities = pcl_arr[:, 3]  
        from sklearn.neighbors import NearestNeighbors
        nbrs = NearestNeighbors(n_neighbors=1).fit(pcl_arr[:, :3])
        _, indices = nbrs.kneighbors(downsampled_positions)
        downsampled_intensities = intensities[indices].reshape(-1, 1)
        downsampled_pcl = np.hstack((downsampled_positions, downsampled_intensities))
    else:
        downsampled_pcl = downsampled_positions
    
    return downsampled_pcl