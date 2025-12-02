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
import torch
import open3d as o3d
from sklearn.neighbors import NearestNeighbors

def gpu_knn_search(pcl_tensor, k=20):
    distances = torch.cdist(pcl_tensor, pcl_tensor)
    distances, indices = torch.topk(distances, k=k, largest=False)
    return distances, indices

def calculate_curvatures_gpu(pcl_tensor, indices):
    neighbors = pcl_tensor[indices]  
    mean = neighbors.mean(dim=1, keepdim=True)  
    centered = neighbors - mean
    covariance = torch.einsum('bij,bik->bjk', centered, centered) / (neighbors.shape[1] - 1)
    eigenvalues = torch.linalg.eigvalsh(covariance)  
    curvatures = eigenvalues[:, 0] / (eigenvalues.sum(dim=1) + 1e-6)
    return curvatures

def calculate_gradients_gpu(pcl_tensor, indices):
    neighbors = pcl_tensor[indices]  
    dz = neighbors[:, :, 2] - pcl_tensor[:, 2].unsqueeze(1)  
    gradients = torch.mean(torch.abs(dz), dim=1)
    return gradients

def extract_features_with_classification_gpu(pcl_arr, curvature_threshold=0.1, gradient_threshold=0.05, voxel_size=0.2, target_num_points=10000):
    pcl_positions = pcl_arr[:, :3]  
    pcl_intensities = pcl_arr[:, 3]  

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pcl_tensor = torch.tensor(pcl_positions, dtype=torch.float32).to(device)

    _, indices = gpu_knn_search(pcl_tensor)

    curvatures = calculate_curvatures_gpu(pcl_tensor, indices)
    gradients = calculate_gradients_gpu(pcl_tensor, indices)

    is_feature = (curvatures > curvature_threshold) | (gradients > gradient_threshold)
    feature_points = pcl_arr[is_feature.cpu().numpy()]  

    feature_curvatures = curvatures[is_feature.cpu().numpy()].cpu().numpy()
    feature_gradients = gradients[is_feature.cpu().numpy()].cpu().numpy()
    feature_points_with_values = np.hstack((feature_points, feature_curvatures.reshape(-1, 1), feature_gradients.reshape(-1, 1)))

    non_feature_points = pcl_arr[~is_feature.cpu().numpy()]  
    if len(non_feature_points) > 0:
        non_feature_positions = non_feature_points[:, :3]
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(non_feature_positions)
        downsampled_pcl = point_cloud.voxel_down_sample(voxel_size)
        downsampled_positions = np.asarray(downsampled_pcl.points)

        downsampled_indices = NearestNeighbors(n_neighbors=1).fit(non_feature_positions).kneighbors(downsampled_positions, return_distance=False)
        average_intensities = np.mean(pcl_intensities[downsampled_indices], axis=1)
        downsampled_points = np.hstack((downsampled_positions, average_intensities.reshape(-1, 1)))

        downsampled_points_extended = np.hstack((downsampled_points, np.zeros((downsampled_points.shape[0], 2))))  
    else:
        downsampled_points_extended = np.empty((0, 6))  

    processed_pcl = np.vstack((feature_points_with_values, downsampled_points_extended))

    if len(processed_pcl) > target_num_points:
        indices = np.random.choice(len(processed_pcl), target_num_points, replace=False)
        processed_pcl = processed_pcl[indices]

    return processed_pcl