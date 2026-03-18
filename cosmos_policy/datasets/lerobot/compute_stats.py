#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import numpy as np
import torch
import einops
from PIL import Image as PILImage
from pathlib import Path


from cosmos_policy.datasets.lerobot.oxe_configs import OXE_DATASET_CONFIGS

def load_image_as_numpy(
    fpath: str | Path, dtype: np.dtype = np.float32, channel_first: bool = True
) -> np.ndarray:
    img = PILImage.open(fpath).convert("RGB")
    img_array = np.array(img, dtype=dtype)
    if channel_first:  # (H, W, C) -> (C, H, W)
        img_array = np.transpose(img_array, (2, 0, 1))
    if np.issubdtype(dtype, np.floating):
        img_array /= 255.0
    return img_array

DEFAULT_QUANTILES = [0.01, 0.10, 0.50, 0.90, 0.99]

class RunningQuantileStats:
    """
    Maintains running statistics for batches of vectors, including mean,
    standard deviation, min, max, and approximate quantiles.

    Statistics are computed per feature dimension and updated incrementally
    as new batches are observed. Quantiles are estimated using histograms,
    which adapt dynamically if the observed data range expands.
    """

    def __init__(self, quantile_list: list[float] | None = None, num_quantile_bins: int = 5000):
        self._count = 0
        self._mean = None
        self._mean_of_squares = None
        self._min = None
        self._max = None
        self._histograms = None
        self._bin_edges = None
        self._num_quantile_bins = num_quantile_bins

        self._quantile_list = quantile_list
        if self._quantile_list is None:
            self._quantile_list = DEFAULT_QUANTILES
        self._quantile_keys = [f"q{int(q * 100):02d}" for q in self._quantile_list]

    def update(self, batch: np.ndarray) -> None:
        """Update the running statistics with a batch of vectors.

        Args:
            batch: An array where all dimensions except the last are batch dimensions.
        """
        batch = batch.reshape(-1, batch.shape[-1])
        num_elements, vector_length = batch.shape

        if self._count == 0:
            self._mean = np.mean(batch, axis=0)
            self._mean_of_squares = np.mean(batch**2, axis=0)
            self._min = np.min(batch, axis=0)
            self._max = np.max(batch, axis=0)
            self._histograms = [np.zeros(self._num_quantile_bins) for _ in range(vector_length)]
            self._bin_edges = [
                np.linspace(self._min[i] - 1e-10, self._max[i] + 1e-10, self._num_quantile_bins + 1)
                for i in range(vector_length)
            ]
        else:
            if vector_length != self._mean.size:
                raise ValueError("The length of new vectors does not match the initialized vector length.")

            new_max = np.max(batch, axis=0)
            new_min = np.min(batch, axis=0)
            max_changed = np.any(new_max > self._max)
            min_changed = np.any(new_min < self._min)
            self._max = np.maximum(self._max, new_max)
            self._min = np.minimum(self._min, new_min)

            if max_changed or min_changed:
                self._adjust_histograms()

        self._count += num_elements

        batch_mean = np.mean(batch, axis=0)
        batch_mean_of_squares = np.mean(batch**2, axis=0)

        # Update running mean and mean of squares
        self._mean += (batch_mean - self._mean) * (num_elements / self._count)
        self._mean_of_squares += (batch_mean_of_squares - self._mean_of_squares) * (
            num_elements / self._count
        )

        self._update_histograms(batch)

    def get_statistics(self) -> dict[str, np.ndarray]:
        """Compute and return the statistics of the vectors processed so far.

        Args:
            quantiles: List of quantiles to compute (e.g., [0.01, 0.10, 0.50, 0.90, 0.99]). If None, no quantiles computed.

        Returns:
            Dictionary containing the computed statistics.
        """
        if self._count < 2:
            raise ValueError("Cannot compute statistics for less than 2 vectors.")

        variance = self._mean_of_squares - self._mean**2

        stddev = np.sqrt(np.maximum(0, variance))

        stats = {
            "min": self._min.copy(),
            "max": self._max.copy(),
            "mean": self._mean.copy(),
            "std": stddev,
            "count": np.array([self._count]),
        }

        quantile_results = self._compute_quantiles()
        for i, q in enumerate(self._quantile_keys):
            stats[q] = quantile_results[i]

        return stats

    def _adjust_histograms(self):
        """Adjust histograms when min or max changes."""
        for i in range(len(self._histograms)):
            old_edges = self._bin_edges[i]
            old_hist = self._histograms[i]

            # Create new edges with small padding to ensure range coverage
            padding = (self._max[i] - self._min[i]) * 1e-10
            new_edges = np.linspace(
                self._min[i] - padding, self._max[i] + padding, self._num_quantile_bins + 1
            )

            # Redistribute existing histogram counts to new bins
            # We need to map each old bin center to the new bins
            old_centers = (old_edges[:-1] + old_edges[1:]) / 2
            new_hist = np.zeros(self._num_quantile_bins)

            for old_center, count in zip(old_centers, old_hist, strict=False):
                if count > 0:
                    # Find which new bin this old center belongs to
                    bin_idx = np.searchsorted(new_edges, old_center) - 1
                    bin_idx = max(0, min(bin_idx, self._num_quantile_bins - 1))
                    new_hist[bin_idx] += count

            self._histograms[i] = new_hist
            self._bin_edges[i] = new_edges

    def _update_histograms(self, batch: np.ndarray) -> None:
        """Update histograms with new vectors."""
        for i in range(batch.shape[1]):
            hist, _ = np.histogram(batch[:, i], bins=self._bin_edges[i])
            self._histograms[i] += hist

    def _compute_quantiles(self) -> list[np.ndarray]:
        """Compute quantiles based on histograms."""
        results = []
        for q in self._quantile_list:
            target_count = q * self._count
            q_values = []

            for hist, edges in zip(self._histograms, self._bin_edges, strict=True):
                q_value = self._compute_single_quantile(hist, edges, target_count)
                q_values.append(q_value)

            results.append(np.array(q_values))
        return results

    def _compute_single_quantile(self, hist: np.ndarray, edges: np.ndarray, target_count: float) -> float:
        """Compute a single quantile value from histogram and bin edges."""
        cumsum = np.cumsum(hist)
        idx = np.searchsorted(cumsum, target_count)

        if idx == 0:
            return edges[0]
        if idx >= len(cumsum):
            return edges[-1]

        # If not edge case, interpolate within the bin
        count_before = cumsum[idx - 1]
        count_in_bin = cumsum[idx] - count_before

        # If no samples in this bin, use the bin edge
        if count_in_bin == 0:
            return edges[idx]

        # Linear interpolation within the bin
        fraction = (target_count - count_before) / count_in_bin
        return edges[idx] + fraction * (edges[idx + 1] - edges[idx])


def estimate_num_samples(
    dataset_len: int, min_num_samples: int = 100, max_num_samples: int = 10_000, power: float = 0.75
) -> int:
    """Heuristic to estimate the number of samples based on dataset size.
    The power controls the sample growth relative to dataset size.
    Lower the power for less number of samples.

    For default arguments, we have:
    - from 1 to ~500, num_samples=100
    - at 1000, num_samples=177
    - at 2000, num_samples=299
    - at 5000, num_samples=594
    - at 10000, num_samples=1000
    - at 20000, num_samples=1681
    """
    if dataset_len < min_num_samples:
        min_num_samples = dataset_len
    return max(min_num_samples, min(int(dataset_len**power), max_num_samples))


def sample_indices(data_len: int) -> list[int]:
    num_samples = estimate_num_samples(data_len)
    return np.round(np.linspace(0, data_len - 1, num_samples)).astype(int).tolist()


def auto_downsample_height_width(img: np.ndarray, target_size: int = 150, max_size_threshold: int = 300):
    _, height, width = img.shape

    if max(width, height) < max_size_threshold:
        # no downsampling needed
        return img

    downsample_factor = int(width / target_size) if width > height else int(height / target_size)
    return img[:, ::downsample_factor, ::downsample_factor]


def sample_images(image_paths: list[str]) -> np.ndarray:
    sampled_indices = sample_indices(len(image_paths))

    images = None
    for i, idx in enumerate(sampled_indices):
        path = image_paths[idx]
        # we load as uint8 to reduce memory usage
        img = load_image_as_numpy(path, dtype=np.uint8, channel_first=True)
        img = auto_downsample_height_width(img)

        if images is None:
            images = np.empty((len(sampled_indices), *img.shape), dtype=np.uint8)

        images[i] = img

    return images


# def get_feature_stats(array: np.ndarray, axis: tuple, keepdims: bool) -> dict[str, np.ndarray]:
#     return {
#         "min": np.min(array, axis=axis, keepdims=keepdims),
#         "max": np.max(array, axis=axis, keepdims=keepdims),
#         "mean": np.mean(array, axis=axis, keepdims=keepdims),
#         "std": np.std(array, axis=axis, keepdims=keepdims),
#         "count": np.array([len(array)]),
#     }

def _prepare_array_for_stats(array: np.ndarray, axis: int | tuple[int, ...] | None) -> tuple[np.ndarray, int]:
    """Prepare array for statistics computation by reshaping according to axis.

    Args:
        array: Input data array
        axis: Axis or axes along which to compute statistics

    Returns:
        Tuple of (reshaped_array, sample_count)
    """
    if axis == (0, 2, 3):  # Image data
        batch_size, channels, height, width = array.shape
        reshaped = array.transpose(0, 2, 3, 1).reshape(-1, channels)
        return reshaped, batch_size

    if axis == 0 or axis == (0,):  # Vector data
        reshaped = array
        if array.ndim == 1:
            reshaped = array.reshape(-1, 1)
        return reshaped, array.shape[0]

    if axis == (1,):  # Feature-wise statistics
        return array.T, array.shape[1]

    if axis is None:  # Global statistics
        reshaped = array.reshape(-1, 1)
        # For backward compatibility, count represents the first dimension size
        return reshaped, array.shape[0] if array.ndim > 0 else 1

    raise ValueError(f"Unsupported axis configuration: {axis}")

def _compute_basic_stats(
    array: np.ndarray, sample_count: int, quantile_list: list[float] | None = None
) -> dict[str, np.ndarray]:
    """Compute basic statistics for arrays with insufficient samples for quantiles.

    Args:
        array: Reshaped array ready for statistics computation
        sample_count: Number of samples represented in the data

    Returns:
        Dictionary with basic statistics and quantiles set to mean values
    """
    if quantile_list is None:
        quantile_list = DEFAULT_QUANTILES
    quantile_list_keys = [f"q{int(q * 100):02d}" for q in quantile_list]

    stats = {
        "min": np.min(array, axis=0),
        "max": np.max(array, axis=0),
        "mean": np.mean(array, axis=0),
        "std": np.std(array, axis=0),
        "count": np.array([sample_count]),
    }

    for q in quantile_list_keys:
        stats[q] = stats["mean"].copy()

    return stats

def _reshape_stats_by_axis(
    stats: dict[str, np.ndarray],
    axis: int | tuple[int, ...] | None,
    keepdims: bool,
    original_shape: tuple[int, ...],
) -> dict[str, np.ndarray]:
    """Reshape all statistics to match NumPy's output conventions.

    Applies consistent reshaping to all statistics (except 'count') based on the
    axis and keepdims parameters. This ensures statistics have the correct shape
    for broadcasting with the original data.

    Args:
        stats: Dictionary of computed statistics
        axis: Axis or axes along which statistics were computed
        keepdims: Whether to keep reduced dimensions as size-1 dimensions
        original_shape: Shape of the original array

    Returns:
        Dictionary with reshaped statistics

    Note:
        The 'count' statistic is never reshaped as it represents metadata
        rather than per-feature statistics.
    """
    if axis == (1,) and not keepdims:
        return stats

    result = {}
    for key, value in stats.items():
        if key == "count":
            result[key] = value
        else:
            result[key] = _reshape_single_stat(value, axis, keepdims, original_shape)

    return result

def _reshape_for_image_stats(value: np.ndarray, keepdims: bool) -> np.ndarray:
    """Reshape statistics for image data (axis=(0,2,3))."""
    if keepdims and value.ndim == 1:
        return value.reshape(1, -1, 1, 1)
    return value


def _reshape_for_vector_stats(
    value: np.ndarray, keepdims: bool, original_shape: tuple[int, ...]
) -> np.ndarray:
    """Reshape statistics for vector data (axis=0 or axis=(0,))."""
    if not keepdims:
        return value

    if len(original_shape) == 1 and value.ndim > 0:
        return value.reshape(1)
    elif len(original_shape) >= 2 and value.ndim == 1:
        return value.reshape(1, -1)
    return value


def _reshape_for_feature_stats(value: np.ndarray, keepdims: bool) -> np.ndarray:
    """Reshape statistics for feature-wise computation (axis=(1,))."""
    if not keepdims:
        return value

    if value.ndim == 0:
        return value.reshape(1, 1)
    elif value.ndim == 1:
        return value.reshape(-1, 1)
    return value


def _reshape_for_global_stats(
    value: np.ndarray, keepdims: bool, original_shape: tuple[int, ...]
) -> np.ndarray | float:
    """Reshape statistics for global reduction (axis=None)."""
    if keepdims:
        target_shape = tuple(1 for _ in original_shape)
        return value.reshape(target_shape)
    # Keep at least 1-D arrays to satisfy validator
    return np.atleast_1d(value)

def _reshape_single_stat(
    value: np.ndarray, axis: int | tuple[int, ...] | None, keepdims: bool, original_shape: tuple[int, ...]
) -> np.ndarray | float:
    """Apply appropriate reshaping to a single statistic array.

    This function transforms statistic arrays to match expected output shapes
    based on the axis configuration and keepdims parameter.

    Args:
        value: The statistic array to reshape
        axis: Axis or axes that were reduced during computation
        keepdims: Whether to maintain reduced dimensions as size-1 dimensions
        original_shape: Shape of the original data before reduction

    Returns:
        Reshaped array following NumPy broadcasting conventions

    """
    if axis == (0, 2, 3):
        return _reshape_for_image_stats(value, keepdims)

    if axis in [0, (0,)]:
        return _reshape_for_vector_stats(value, keepdims, original_shape)

    if axis == (1,):
        return _reshape_for_feature_stats(value, keepdims)

    if axis is None:
        return _reshape_for_global_stats(value, keepdims, original_shape)

    return value



def get_feature_stats(
    array: np.ndarray,
    axis: int | tuple[int, ...] | None,
    keepdims: bool,
    quantile_list: list[float] | None = None,
) -> dict[str, np.ndarray]:
    """Compute comprehensive statistics for array features along specified axes.

    This function calculates min, max, mean, std, and quantiles (1%, 10%, 50%, 90%, 99%)
    for the input array along the specified axes. It handles different data layouts:
    - Image data: axis=(0,2,3) computes per-channel statistics
    - Vector data: axis=0 computes per-feature statistics
    - Feature-wise: axis=1 computes statistics across features
    - Global: axis=None computes statistics over entire array

    Args:
        array: Input data array with shape appropriate for the specified axis
        axis: Axis or axes along which to compute statistics
            - (0, 2, 3): For image data (batch, channels, height, width)
            - 0 or (0,): For vector/tabular data (samples, features)
            - (1,): For computing across features
            - None: For global statistics over entire array
        keepdims: If True, reduced axes are kept as dimensions with size 1

    Returns:
        Dictionary containing:
            - 'min': Minimum values
            - 'max': Maximum values
            - 'mean': Mean values
            - 'std': Standard deviation
            - 'count': Number of samples (always shape (1,))
            - 'q01', 'q10', 'q50', 'q90', 'q99': Quantile values

    """
    if quantile_list is None:
        quantile_list = DEFAULT_QUANTILES

    original_shape = array.shape
    reshaped, sample_count = _prepare_array_for_stats(array, axis)

    if reshaped.shape[0] < 2:
        stats = _compute_basic_stats(reshaped, sample_count, quantile_list)
    else:
        running_stats = RunningQuantileStats()
        running_stats.update(reshaped)
        stats = running_stats.get_statistics()
        stats["count"] = np.array([sample_count])

    stats = _reshape_stats_by_axis(stats, axis, keepdims, original_shape)
    return stats


def compute_episode_stats(episode_data: dict[str, list[str] | np.ndarray], features: dict) -> dict:
    ep_stats = {}
    for key, data in episode_data.items():
        if features[key]["dtype"] == "string":
            continue  # HACK: we should receive np.arrays of strings
        elif features[key]["dtype"] in ["image", "video"]:
            ep_ft_array = sample_images(data)  # data is a list of image paths
            axes_to_reduce = (0, 2, 3)  # keep channel dim
            keepdims = True
        else:
            ep_ft_array = data  # data is already a np.ndarray
            axes_to_reduce = 0  # compute stats over the first axis
            keepdims = data.ndim == 1  # keep as np.array

        # ep_stats[key] = get_feature_stats(ep_ft_array, axis=axes_to_reduce, keepdims=keepdims)

        # # finally, we normalize and remove batch dim for images
        # if features[key]["dtype"] in ["image", "video"]:
        #     ep_stats[key] = {
        #         k: v if k == "count" else np.squeeze(v / 255.0, axis=0) for k, v in ep_stats[key].items()
        #     }

    return ep_stats


def _assert_type_and_shape(stats_list: list[dict[str, dict]]):
    for i in range(len(stats_list)):
        for fkey in stats_list[i]:
            for k, v in stats_list[i][fkey].items():
                if not isinstance(v, np.ndarray):
                    raise ValueError(
                        f"Stats must be composed of numpy array, but key '{k}' of feature '{fkey}' is of type '{type(v)}' instead."
                    )
                if v.ndim == 0:
                    raise ValueError("Number of dimensions must be at least 1, and is 0 instead.")
                if k == "count" and v.shape != (1,):
                    raise ValueError(f"Shape of 'count' must be (1), but is {v.shape} instead.")
                if "image" in fkey and k != "count" and v.shape != (3, 1, 1):
                    raise ValueError(f"Shape of '{k}' must be (3,1,1), but is {v.shape} instead.")


def aggregate_feature_stats(stats_ft_list: list[dict[str, dict]]) -> dict[str, dict[str, np.ndarray]]:
    """Aggregates stats for a single feature."""
    means = np.stack([s["mean"] for s in stats_ft_list])
    variances = np.stack([s["std"] ** 2 for s in stats_ft_list])
    counts = np.stack([s["count"] for s in stats_ft_list])
    total_count = counts.sum(axis=0)

    # Prepare weighted mean by matching number of dimensions
    while counts.ndim < means.ndim:
        counts = np.expand_dims(counts, axis=-1)

    # Compute the weighted mean
    weighted_means = means * counts
    total_mean = weighted_means.sum(axis=0) / total_count

    # Compute the variance using the parallel algorithm
    delta_means = means - total_mean
    weighted_variances = (variances + delta_means**2) * counts
    total_variance = weighted_variances.sum(axis=0) / total_count

    aggregated = {
        "min": np.min(np.stack([s["min"] for s in stats_ft_list]), axis=0),
        "max": np.max(np.stack([s["max"] for s in stats_ft_list]), axis=0),
        "mean": total_mean,
        "std": np.sqrt(total_variance),
        "count": total_count,
    }

    if stats_ft_list:
        quantile_keys = [k for k in stats_ft_list[0] if k.startswith("q") and k[1:].isdigit()]

        for q_key in quantile_keys:
            if all(q_key in s for s in stats_ft_list):
                quantile_values = np.stack([s[q_key] for s in stats_ft_list])
                # print(quantile_values)
                weighted_quantiles = quantile_values * counts
                aggregated[q_key] = weighted_quantiles.sum(axis=0) / total_count
                # if q_key == "q01":
                #     aggregated[q_key] = quantile_values.min(axis=0)
                # elif q_key == "q99":
                #     aggregated[q_key] = quantile_values.max(axis=0)
                # else:
                #     # 中间的 quantile，比如 q50，可以用平均
                #     aggregated[q_key] = quantile_values.mean(axis=0)

    return aggregated

def aggregate_stats_v2(stats_list: list[dict[str, dict]], max_dim = 32) -> dict[str, dict[str, np.ndarray]]:
    """Aggregate stats from multiple compute_stats outputs into a single set of stats.

    The final stats will have the union of all data keys from each of the stats dicts.

    For instance:
    - new_min = min(min_dataset_0, min_dataset_1, ...)
    - new_max = max(max_dataset_0, max_dataset_1, ...)
    - new_mean = (mean of all data, weighted by counts)
    - new_std = (std of all data)
    """

    _assert_type_and_shape(stats_list)

    data_keys = {key for stats in stats_list for key in stats}
    aggregated_stats = {key: {} for key in data_keys}

    for key in data_keys:
        stats_with_key = [stats[key] for stats in stats_list if key in stats]
        aggregated_stats[key] = aggregate_feature_stats(stats_with_key)

    return aggregated_stats

def aggregate_stats(stats_list: list[dict[str, dict]], max_dims = 32) -> dict[str, dict[str, np.ndarray]]:
    """Aggregate stats from multiple compute_stats outputs into a single set of stats.

    The final stats will have the union of all data keys from each of the stats dicts.

    For instance:
    - new_min = min(min_dataset_0, min_dataset_1, ...)
    - new_max = max(max_dataset_0, max_dataset_1, ...)
    - new_mean = (mean of all data, weighted by counts)
    - new_std = (std of all data)
    """

    _assert_type_and_shape(stats_list)

    data_keys = {key for stats in stats_list for key in stats}
    aggregated_stats = {key: {} for key in data_keys}

    for key in data_keys:
        stats_with_key = [stats[key] for stats in stats_list if key in stats]
        # pad mean, std, q01, q99, max, min
        # for dataset without ego_dex
        if key in ["action", "observation.state"]:
            pad_stats_with_key = []
            max_dim = max_dims[key]
            for stats in stats_with_key: # for different dataset
                pad_stats = {}
                pad_len = max_dim - len(stats["mean"])
                if pad_len <= 0:
                    pad_stats_with_key.append(stats)
                    continue
                # np.pad(数组, (左补数量, 右补数量), mode="constant", constant_values=填充值)
                pad_stats["mean"] = np.pad(stats["mean"], (0, pad_len), mode="constant", constant_values=0)
                pad_stats["std"] = np.pad(stats["std"], (0, pad_len), mode="constant", constant_values=1)
                pad_stats["max"] = np.pad(stats["max"], (0, pad_len), mode="constant", constant_values=1)
                pad_stats["min"] = np.pad(stats["min"], (0, pad_len), mode="constant", constant_values=-1)
                pad_stats["q01"] = np.pad(stats["q01"], (0, pad_len), mode="constant", constant_values=-1)
                pad_stats["q10"] = np.pad(stats["q10"], (0, pad_len), mode="constant", constant_values=-0.5)
                pad_stats["q50"] = np.pad(stats["q50"], (0, pad_len), mode="constant", constant_values=0)
                pad_stats["q90"] = np.pad(stats["q90"], (0, pad_len), mode="constant", constant_values=0.5)
                pad_stats["q99"] = np.pad(stats["q99"], (0, pad_len), mode="constant", constant_values=1)
                pad_stats["count"] = stats["count"]
                pad_stats_with_key.append(pad_stats)
        else:
            pad_stats_with_key = stats_with_key
        aggregated_stats[key] = aggregate_feature_stats(pad_stats_with_key)
        for k, v in aggregated_stats[key].items():
            if isinstance(aggregated_stats[key][k], np.ndarray):
                aggregated_stats[key][k] = torch.from_numpy(aggregated_stats[key][k])
        # print(key, type(aggregated_stats[key]))

    return aggregated_stats


def update_rh20t_stats(stats, max_dim, key, stat_key, constant_v):
    if key == "action":
        prefix = 6
        force_start = 44
        force_end = 44 + 6
    else:
        prefix = 7
        force_start = 46
        force_end = 46 + 6
    stats_value = stats[stat_key]
    xyz_rpy_mean = stats_value[:prefix]
    gripper_mean = stats_value[-2:-1]
    force_mean = stats_value[prefix:prefix + 6]
    new_stats = np.ones(max_dim) * constant_v
    # print(new_stats.shape, force_mean.shape, stats_value.shape)
    new_stats[:prefix] = xyz_rpy_mean
    new_stats[prefix:prefix + 1] = gripper_mean
    new_stats[force_start:force_end] = force_mean
    return new_stats
    
    

def aggregate_stats_with_game(stats_list_with_name, max_dim = 32) -> dict[str, dict[str, np.ndarray]]:
    """Aggregate stats from multiple compute_stats outputs into a single set of stats.

    The final stats will have the union of all data keys from each of the stats dicts.

    For instance:
    - new_min = min(min_dataset_0, min_dataset_1, ...)
    - new_max = max(max_dataset_0, max_dataset_1, ...)
    - new_mean = (mean of all data, weighted by counts)
    - new_std = (std of all data)
    """

    stats_list = stats_list_with_name["stats"]
    dataset_names = stats_list_with_name["dataset_names"]
    _assert_type_and_shape(stats_list)

    data_keys = {key for stats in stats_list for key in stats} # action, state, timestamp
    aggregated_stats = {key: {} for key in data_keys}
    
    process_keys = ["action", "observation.state"]
    
    for i in range(len(stats_list)):
        stats_with_key = stats_list[i] # dataset stats from all keys
        dataset_name = dataset_names[i]
        for key in process_keys:
            stats = stats_with_key[key] # key stats
            if "game" in dataset_name:
                if key == "action":
                    pad_len = 44 + 6 # dual robot arm + force(single arm)
                else:
                    pad_len = 46 + 6
                stats["mean"] = np.pad(stats["mean"], (pad_len, 0),  # 前面补 44 个，后面不补
                    mode="constant",constant_values=0)
                stats["std"] = np.pad(stats["std"], (pad_len, 0),  # 前面补 44 个，后面不补
                    mode="constant",constant_values=1)
                stats["max"] = np.pad(stats["max"], (pad_len, 0),  # 前面补 44 个，后面不补
                    mode="constant",constant_values=0)
                stats["min"] = np.pad(stats["min"], (pad_len, 0),  # 前面补 44 个，后面不补
                    mode="constant",constant_values=0)
                stats["q01"] = np.pad(stats["q01"], (pad_len, 0),  # 前面补 44 个，后面不补
                    mode="constant",constant_values=-1)
                stats["q10"] = np.pad(stats["q10"], (pad_len, 0),  # 前面补 44 个，后面不补
                    mode="constant",constant_values=-0.5)
                stats["q50"] = np.pad(stats["q50"], (pad_len, 0),  # 前面补 44 个，后面不补
                    mode="constant",constant_values=0)
                stats["q90"] = np.pad(stats["q90"], (pad_len, 0),  # 前面补 44 个，后面不补
                    mode="constant",constant_values=0.5)
                stats["q99"] = np.pad(stats["q99"], (pad_len, 0),  # 前面补 44 个，后面不补
                    mode="constant",constant_values=1)
            elif "rh20t" in dataset_name:
                stats["mean"] = update_rh20t_stats(stats, max_dim, key, "mean", constant_v = 0)
                stats["std"] = update_rh20t_stats(stats, max_dim, key, "std", constant_v = 1)
                stats["max"] = update_rh20t_stats(stats, max_dim, key, "max", constant_v = 0)
                stats["min"] = update_rh20t_stats(stats, max_dim, key, "min", constant_v = 0)
                stats["q01"] = update_rh20t_stats(stats, max_dim, key, "q01", constant_v = -1)
                stats["q10"] = update_rh20t_stats(stats, max_dim, key, "q10", constant_v = -0.5)
                stats["q50"] = update_rh20t_stats(stats, max_dim, key, "q50", constant_v = 0)
                stats["q90"] = update_rh20t_stats(stats, max_dim, key, "q90", constant_v = 0.5)
                stats["q99"] = update_rh20t_stats(stats, max_dim, key, "q99", constant_v = 1)
            
            stats_list[i][key] = stats

    for key in data_keys:
        # stats: ["action":["mean":, "std": , ], "state":["mean":, "std":, ]]
        stats_with_key = [stats[key] for stats in stats_list if key in stats] # contains all dataset
        # pad mean, std, q01, q99, max, min
        # for dataset without ego_dex
        if key in ["action", "observation.state"]:
            pad_stats_with_key = []
            for stats in stats_with_key:
                pad_stats = {}
                pad_len = max_dim - len(stats["mean"])
                # np.pad(数组, (左补数量, 右补数量), mode="constant", constant_values=填充值)
                pad_stats["mean"] = np.pad(stats["mean"], (0, pad_len), mode="constant", constant_values=0)
                pad_stats["std"] = np.pad(stats["std"], (0, pad_len), mode="constant", constant_values=1)
                pad_stats["max"] = np.pad(stats["max"], (0, pad_len), mode="constant", constant_values=0)
                pad_stats["min"] = np.pad(stats["min"], (0, pad_len), mode="constant", constant_values=0)
                pad_stats["q01"] = np.pad(stats["q01"], (0, pad_len), mode="constant", constant_values=-1)
                pad_stats["q10"] = np.pad(stats["q10"], (0, pad_len), mode="constant", constant_values=-0.5)
                pad_stats["q50"] = np.pad(stats["q50"], (0, pad_len), mode="constant", constant_values=0)
                pad_stats["q90"] = np.pad(stats["q90"], (0, pad_len), mode="constant", constant_values=0.5)
                pad_stats["q99"] = np.pad(stats["q99"], (0, pad_len), mode="constant", constant_values=1)
                pad_stats["count"] = stats["count"]
                pad_stats_with_key.append(pad_stats)
        else:
            pad_stats_with_key = stats_with_key
        aggregated_stats[key] = aggregate_feature_stats(pad_stats_with_key)
        for k, v in aggregated_stats[key].items():
            if isinstance(aggregated_stats[key][k], np.ndarray):
                aggregated_stats[key][k] = torch.from_numpy(aggregated_stats[key][k])
        # print(key, type(aggregated_stats[key]))

    return aggregated_stats

    