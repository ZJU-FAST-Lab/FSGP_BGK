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
import cupy as cp
from cupyx.scipy.ndimage import convolve as cp_convolve
from scipy.ndimage import gaussian_filter, median_filter
import time
from collections import deque

class TraversabilityAnalyzerWithBGK_GPU:
    def __init__(self, resolution=0.2, x_length=10, y_length=10, time_window=600.0, max_history_frames=1):
        self.resolution = resolution
        self.x_length = x_length
        self.y_length = y_length
        self.grid_size = int(x_length / resolution)
        self.map_size = self.grid_size**2

        self.global_grid_offset = [0, 0]  
        self.historical_observations = {}  # {cell: deque of (value, variance, count, timestamp)}
        self.time_window = time_window  
        self.max_history_frames = max_history_frames  

    @staticmethod
    def normalize_attribute(attribute, global_min, global_max):
        if global_max == global_min:
            return np.zeros_like(attribute)  
        attribute = np.asarray(attribute, dtype=np.float64)
        return np.clip((attribute - global_min) / (global_max - global_min + 1e-10), 0, 1)

    def preprocess_data(self, data, method="gaussian", **kwargs):
        if method == "gaussian":
            sigma = kwargs.get("sigma", 1.0)
            return gaussian_filter(data, sigma=sigma)
        elif method == "median":
            size = kwargs.get("size", 3)
            return median_filter(data, size=size)
        else:
            raise ValueError("Unsupported filtering method")

    def calculate_slope(self, grad_mean, max_slope, min_slope, test_slope=False):
        grad_mean = self.preprocess_data(grad_mean, method="gaussian", sigma=1.0)
        slope = np.linalg.norm(grad_mean, axis=1)
        if test_slope:
            print(max(slope), min(slope))
        return self.normalize_attribute(slope, min_slope, max_slope)

    def calculate_flatness_entropy(self, height_map, max_flatness, min_flatness, test_flatness=False):
        p = self.preprocess_data(height_map, method="gaussian", sigma=0.5)
        if test_flatness:
            print(max(p), min(p))
        return self.normalize_attribute(p, min_flatness, max_flatness)

    def calculate_step_height_topology(self, height_map, max_height, min_height, test_height=False):
        p = self.preprocess_data(height_map, method="gaussian", sigma=0.5)
        if test_height:
            print(max(p), min(p))
        return self.normalize_attribute(p, min_height, max_height)

    def calculate_uncertainty_information_gain(self, preds, prior_entropy, max_uncertainty, min_uncertainty, test_uncertainty=False):
        preds = self.preprocess_data(preds, method="gaussian", sigma=0.5)
        posterior_entropy = -preds * np.log2(preds + 1e-10)  
        info_gain = prior_entropy - posterior_entropy
        if test_uncertainty:
            print(max(info_gain), min(info_gain))
        return self.normalize_attribute(info_gain, min_uncertainty, max_uncertainty)

    def update_global_grid_offset(self, new_position):
        x_offset = int(new_position[0] // self.resolution)
        y_offset = int(new_position[1] // self.resolution)

        delta_x = x_offset - self.global_grid_offset[0]
        delta_y = y_offset - self.global_grid_offset[1]

        if abs(delta_x) >= self.grid_size or abs(delta_y) >= self.grid_size:
            self.historical_observations.clear()
        else:
            to_delete = [
                cell for cell in self.historical_observations
                if not (0 <= cell[0] - delta_x < self.grid_size and 0 <= cell[1] - delta_y < self.grid_size)
            ]
            for cell in to_delete:
                del self.historical_observations[cell]

        self.global_grid_offset = [x_offset, y_offset]

    def update_historical_observations(self, new_observations):
        current_time = time.time()

        for cell, (new_value, new_variance) in new_observations.items():
            if cell not in self.historical_observations:
                self.historical_observations[cell] = deque(maxlen=self.max_history_frames)

            self.historical_observations[cell].append((new_value, new_variance, current_time))

    def clean_old_observations(self):
        current_time = time.time()
        for cell in list(self.historical_observations.keys()):
            self.historical_observations[cell] = deque(
                [(v, var, t) for v, var, t in self.historical_observations[cell] if current_time - t <= self.time_window],
                maxlen=self.max_history_frames
            )
            if not self.historical_observations[cell]:  
                del self.historical_observations[cell]

    def spatial_temporal_bgk_inference(self, grid_cells, kernel_function, variance_weight=0.1, radius=2.0):
        observed_coords = []
        observed_values = []
        observed_variances = []
        observed_weights = []

        current_time = time.time()

        for cell, history in self.historical_observations.items():
            for value, variance, timestamp in history:
                time_weight = np.exp(-(current_time - timestamp) / self.time_window)  
                observed_coords.append(cell)
                observed_values.append(value)
                observed_variances.append(variance)
                observed_weights.append(time_weight)

        if not observed_coords:
            return cp.zeros(len(grid_cells), dtype=cp.float32).get()

        observed_coords = cp.array(observed_coords, dtype=cp.float32)
        observed_values = cp.array(observed_values, dtype=cp.float32)
        observed_variances = cp.array(observed_variances, dtype=cp.float32)
        observed_weights = cp.array(observed_weights, dtype=cp.float32)

        grid_cells = cp.array(grid_cells, dtype=cp.float32)
        fused_values = cp.zeros(len(grid_cells), dtype=cp.float32)

        dists = cp.linalg.norm(grid_cells[:, None, :] - observed_coords[None, :, :], axis=2)
        mask = dists <= radius
        weights = kernel_function(dists * mask) / (observed_variances[None, :] + variance_weight)
        weights *= observed_weights  
        weights_sum = cp.maximum(weights.sum(axis=1, keepdims=True), 1e-10)

        fused_values = (weights @ observed_values) / weights_sum.flatten()
        return fused_values.get()

    def calculate_traversability(self, mean, slope, flatness, step_height, uncertainties, position, w_slope, w_flatness, w_step_height, bgk_threshold):
        self.update_global_grid_offset(position)
        map_size = int(cp.sqrt(len(slope)))
        grid_cells = [(i, j) for i in range(map_size) for j in range(map_size)]

        def kernel_function(distance, l=0.1):
            return cp.exp(-distance**2 / (2 * l**2))

        traversability_pre = (slope * w_slope + flatness * w_flatness + step_height * w_step_height)

        mean_variance = cp.var(mean)

        if mean_variance > bgk_threshold:
            new_observations = {cell: (traversability_pre[idx], uncertainties[idx]) for idx, cell in enumerate(grid_cells)}

            self.update_historical_observations(new_observations)

            traversability = self.spatial_temporal_bgk_inference(grid_cells, kernel_function, variance_weight=0.1)

            min_num, max_num = cp.min(traversability), cp.max(traversability)
            range_val = max_num - min_num if max_num != min_num else 1e-10
            traversability = (traversability - min_num) / range_val
        else:
            traversability = 1 - traversability_pre

        return traversability


if __name__ == "__main__":
    analyzer = TraversabilityAnalyzerWithBGK_GPU()

    grad_mean = cp.random.rand(2500, 2) * 30
    height_map = cp.random.rand(2500)
    uncertainties = cp.random.rand(2500) * 0.2
    height = cp.random.rand(2500)

    slope = cp.linalg.norm(grad_mean, axis=1)
    flatness = cp.abs(cp_convolve(height_map.reshape(50, 50), cp.ones((3, 3)) / 9).flatten())
    step_height = cp.abs(cp_convolve(height_map.reshape(50, 50), cp.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])).flatten())

    assert slope.shape == flatness.shape == step_height.shape == uncertainties.shape == height.shape, "Feature shapes are inconsistent!"

    traversability = analyzer.calculate_traversability(slope, flatness, step_height, uncertainties, height, w_slope=0.5, w_flatness=0.3, w_step_height=0.2, bgk_threshold=0.1)

    print(f"Traversability: {traversability.get()}")