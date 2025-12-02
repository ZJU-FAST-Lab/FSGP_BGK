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
from sklearn.neighbors import NearestNeighbors
import traversability_lib.choose_point as choose_point
from sklearn.decomposition import PCA
import torch
import gpytorch
from traversability_lib.sgp_model import SGPModel
from traversability_lib.traversability import TraversabilityAnalyzerWithBGK_GPU
import yaml
from traversability_lib import point_cloud_tool
import time

class TraversabilityAnalyzer:
    def __init__(self, config_path="../config/params.yaml"):
        self.load_config(config_path)
        self.initialize_components()

    def load_config(self, config_path):
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)

        self.curvature_threshold = config["curvature_threshold"]
        self.gradient_threshold = config["gradient_threshold"]
        self.key_voxel_size = config["key_voxel_size"]
        self.inducing_points = config["inducing_points"]
        self.lengthscale = config["lengthscale"]
        self.alpha = config["alpha"]

        self.resolution = config["resolution"]
        self.x_length = config["x_length"]
        self.y_length = config["y_length"]

        self.max_slope = config["max_slope"]
        self.min_slope = config["min_slope"]
        self.test_slope = config["test_slope"]

        self.max_flatness = config["max_flatness"]
        self.min_flatness = config["min_flatness"]
        self.test_flatness = config["test_flatness"]

        self.max_height = config["max_height"]
        self.min_height = config["min_height"]
        self.test_height = config["test_height"]

        self.max_uncertainty = config["max_uncertainty"]
        self.min_uncertainty = config["min_uncertainty"]
        self.test_uncertainty = config["test_uncertainty"]

        self.w_slope = config["w_slope"]
        self.w_flatness = config["w_flatness"]
        self.w_step_height = config["w_step_height"]
        
        self.time_window = config["time_window"]
        self.max_history_frames = config["max_history_frames"]
        
        self.i_num = config["i_num"]
        self.base_height = config["base_height"]
        self.bgk_threshold = config["bgk_threshold"]
        self.downsampl_voxel_size = config["downsampl_voxel_size"]
        self.open_pca=config["open_pca"]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def initialize_components(self):
        self.pca = PCA(n_components=4)
        self.grid = None
        self.Xs = None
        self.Ys = None
        self.Zs = None
        self.curvatures = None
        self.gradients = None

        self.analyzer = TraversabilityAnalyzerWithBGK_GPU(
            self.resolution, self.x_length, self.y_length, self.time_window, self.max_history_frames
        )
        
        self.mean = None
        self.grad_mean = None
        self.slope = None
        self.flatness = None
        self.step_height = None
        self.uncertainty = None
        self.traversability = None
        
        self.pose = None
        self.traversability_dict = None
        self.kd_tree = None
        self.data_dict = None

    def fake_intensity(self, pcl, num_points):
        num_original_points = pcl.shape[0]
        return np.hstack([pcl, np.ones((num_original_points, 1))])

    def sampling_grid(self):
        x_range = self.x_length / 2
        y_range = self.y_length / 2
        x_s = np.arange(-x_range, x_range, self.resolution, dtype='float32')
        y_s = np.arange(-y_range, y_range, self.resolution, dtype='float32')

        grid = np.array(np.meshgrid(x_s, y_s)).T.reshape(-1, 2)
        X_train = np.column_stack((self.Xs, self.Ys))
        curvatures_train = self.curvatures
        gradients_train = self.gradients

        knn = NearestNeighbors(n_neighbors=5, algorithm='auto').fit(X_train)
        distances, indices = knn.kneighbors(grid)

        weights = 1 / (distances + 1e-6)
        weights /= np.sum(weights, axis=1, keepdims=True)

        curvatures_pred = np.sum(weights[:, :, None] * curvatures_train[indices], axis=1)
        gradients_pred = np.sum(weights[:, :, None] * gradients_train[indices], axis=1)

        self.grid = np.column_stack((grid, curvatures_pred, gradients_pred))

    def filter_low_uncertainty_data(self, mean, var, grad_mean, grid, threshold=1.35):
        low_uncertainty_indices = np.where(var < threshold)[0]
        filtered_mean = mean[low_uncertainty_indices]
        filtered_var = var[low_uncertainty_indices]
        filtered_grad_mean = grad_mean[low_uncertainty_indices]
        filtered_grid = grid[low_uncertainty_indices]
        
        return filtered_mean, filtered_var, filtered_grad_mean, filtered_grid

    def generate_robot_points(self, l, w):
        x = np.linspace(-l/2, l/2, num=self.i_num)
        y = np.linspace(-w/2, w/2, num=self.i_num)
        x, y = np.meshgrid(x, y)
        x = x.flatten()
        y = y.flatten()
        z = np.full_like(x, self.base_height)
        i = np.full_like(x, 0)
        c = np.full_like(x, 0)
        g = np.full_like(x, 0)
        
        return np.column_stack((x, y, z, i, c, g))

    def generate_local_traversability_map(self, pose, transformed_points):
        self.pose = pose
        start_time = time.time()
        expanded_pcl = self.fake_intensity(transformed_points, 5000)
        expanded_pcl = point_cloud_tool.voxel_downsample(expanded_pcl, voxel_size=self.downsampl_voxel_size)

        key_points = choose_point.extract_features_with_classification_gpu(
            expanded_pcl,
            curvature_threshold=self.curvature_threshold,
            gradient_threshold=self.gradient_threshold,
            voxel_size=self.key_voxel_size,
            target_num_points=self.inducing_points
        )
       
        self.keypoints = key_points
        robot_points = self.generate_robot_points(self.x_length, self.y_length)
        key_points = np.vstack((key_points, robot_points))
        
        self.Xs = key_points[:, 0].reshape(-1, 1)
        self.Ys = key_points[:, 1].reshape(-1, 1)
        self.Zs = key_points[:, 2].reshape(-1, 1)
        self.curvatures = key_points[:, 4].reshape(-1, 1)
        self.gradients = key_points[:, 5].reshape(-1, 1)
        
        start_time = time.time()
        data = np.column_stack((self.Xs, self.Ys, self.curvatures, self.gradients))
        grid_train=None
        if self.open_pca:
            grid_train=self.pca.fit_transform(data)
        else:
            grid_train=data
        d_in = torch.tensor(grid_train, dtype=torch.float32, device=self.device)
        d_out = torch.tensor(self.Zs, dtype=torch.float32, device=self.device).squeeze()
        
        likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.GreaterThan(1e-4))
        sgp_model = SGPModel(d_in, d_out, likelihood, self.inducing_points, self.lengthscale, self.alpha).to(self.device)
        
        start_time = time.time()
        sgp_model.train()
        likelihood.train()
        optimizer = torch.optim.AdamW(sgp_model.parameters(), lr=0.01)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, sgp_model)

        optimizer.zero_grad()
        output = sgp_model(d_in)
        loss = -mll(output, d_out).mean()
        loss.backward()
        optimizer.step()
        
        sgp_model.eval()
        likelihood.eval()
        
        self.sampling_grid()
        grid_test=None
        if self.open_pca:
            grid_test=self.pca.fit_transform(self.grid)
        else:
            grid_test=self.grid
        Xtest_tensor = torch.tensor(grid_test, dtype=torch.float32, requires_grad=True).to(sgp_model.device)
        start_time = time.time()
        preds = sgp_model.likelihood(sgp_model(Xtest_tensor))
        mean = preds.mean.detach().cpu().numpy()
        var = preds.variance.detach().cpu().numpy()
        grad_mean = torch.autograd.grad(preds.mean.sum(), Xtest_tensor, create_graph=True)[0].detach().cpu().numpy()
        
        filtered_mean, filtered_var, filtered_grad_mean, filter_grid = self.filter_low_uncertainty_data(mean, var, grad_mean, self.grid)
        
        start_time = time.time()
        self.mean = mean
        self.grad_mean = filtered_grad_mean
        self.slope = self.analyzer.calculate_slope(filtered_grad_mean, self.max_slope, self.min_slope, self.test_slope)
        self.flatness = self.analyzer.calculate_flatness_entropy(np.array(filter_grid[:, 2]), self.max_flatness, self.min_flatness, self.test_flatness)
        self.step_height = self.analyzer.calculate_step_height_topology(np.array(filter_grid[:, 3]), self.max_height, self.min_height, self.test_height)
        self.uncertainty = self.analyzer.calculate_uncertainty_information_gain(filtered_var, 1, self.max_uncertainty, self.min_uncertainty, self.test_uncertainty)
        current_position = (pose[0], pose[1])
        
        start_time = time.time()
        self.traversability = 1.0 - self.analyzer.calculate_traversability(self.mean, self.slope, self.flatness, self.step_height, self.uncertainty, current_position, self.w_slope, self.w_flatness, self.w_step_height, self.bgk_threshold)

    def update_map(self, pose, local_pointcloud):
        local_pointcloud = np.array(local_pointcloud)
        self.generate_local_traversability_map(pose, local_pointcloud)