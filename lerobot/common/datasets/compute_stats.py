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
from copy import deepcopy
from math import ceil
from pathlib import Path

import einops
import numpy as np
import pandas as pd
import torch
import tqdm


def get_stats_einops_patterns(dataset, num_workers=0):
    """These einops patterns will be used to aggregate batches and compute statistics.

    Note: We assume the images are in channel first format
    """

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_size=2,
        shuffle=False,
    )
    batch = next(iter(dataloader))

    stats_patterns = {}

    for key in dataset.features:
        # sanity check that tensors are not float64
        assert batch[key].dtype != torch.float64

        # if isinstance(feats_type, (VideoFrame, Image)):
        if key in dataset.meta.camera_keys:
            # sanity check that images are channel first
            _, c, h, w = batch[key].shape
            assert c < h and c < w, f"expect channel first images, but instead {batch[key].shape}"

            # sanity check that images are float32 in range [0,1]
            assert batch[key].dtype == torch.float32, f"expect torch.float32, but instead {batch[key].dtype=}"
            assert batch[key].max() <= 1, f"expect pixels lower than 1, but instead {batch[key].max()=}"
            assert batch[key].min() >= 0, f"expect pixels greater than 1, but instead {batch[key].min()=}"

            stats_patterns[key] = "b c h w -> c 1 1"
        elif batch[key].ndim == 2:
            # Check if this is a flattened point cloud
            if "pointcloud" in key:
                # For point clouds, we want to compute statistics per coordinate (x,y,z)
                # Assuming the flattened array has shape (batch_size, N*3) where N is number of points
                # We reshape to (batch, N, 3) and compute stats per coordinate
                stats_patterns[key] = "b n c -> c"
            else:
                stats_patterns[key] = "b c -> c "
        elif batch[key].ndim == 1:
            # Check if this is a flattened point cloud
            if "pointcloud" in key:
                # For point clouds, we want to compute statistics per coordinate (x,y,z)
                # Assuming the flattened array has shape (batch_size, N*3) where N is number of points
                # We reshape to (batch_size, N, 3) and compute stats per coordinate
                stats_patterns[key] = "b n c -> c"
            else:
                stats_patterns[key] = "b -> 1"
        else:
            raise ValueError(f"{key}, {batch[key].shape}")

    return stats_patterns


def compute_stats_from_parquet(dataset_root: str | Path) -> dict[str, dict[str, torch.Tensor]]:
    """Compute stats for low-dim features directly from parquet files (no video decoding).

    This is much faster than compute_stats() which iterates through the full dataset
    including video frame decoding. Matches the approach used by gr00t's generate_stats().

    Computes: mean, std, min, max, q01, q99 for each float feature in the parquet data.

    Args:
        dataset_root: Path to the dataset root directory containing data/*/*.parquet and meta/info.json.

    Returns:
        Dict mapping feature names to dicts of stat tensors.
    """
    import json

    dataset_root = Path(dataset_root)

    # Identify low-dim (float) features from info.json
    with open(dataset_root / "meta" / "info.json", "r") as f:
        info = json.load(f)
    lowdim_features = [
        feat for feat, spec in info["features"].items()
        if "float" in spec.get("dtype", "")
    ]

    # Load all parquet files
    parquet_paths = sorted(dataset_root.glob("data/*/*.parquet"))
    all_data = pd.concat(
        [pd.read_parquet(p) for p in tqdm.tqdm(parquet_paths, desc="Loading parquet files")],
        axis=0,
    )

    stats = {}
    for feat in lowdim_features:
        if feat not in all_data.columns:
            continue
        np_data = np.vstack([np.asarray(x, dtype=np.float32) for x in all_data[feat]])
        stats[feat] = {
            "mean": torch.from_numpy(np.mean(np_data, axis=0).astype(np.float32)),
            "std": torch.from_numpy(np.std(np_data, axis=0).astype(np.float32)),
            "min": torch.from_numpy(np.min(np_data, axis=0).astype(np.float32)),
            "max": torch.from_numpy(np.max(np_data, axis=0).astype(np.float32)),
            "q01": torch.from_numpy(np.quantile(np_data, 0.01, axis=0).astype(np.float32)),
            "q99": torch.from_numpy(np.quantile(np_data, 0.99, axis=0).astype(np.float32)),
        }
    return stats


def compute_relative_stats_from_parquet(
    dataset_root: str | Path,
    relative_action_keys: list[str] | None = None,
    delta_indices: list[int] | None = None,
) -> dict[str, dict[str, list]]:
    """Compute relative action stats from parquet files. Matches gr00t's generate_rel_stats().

    For each timestep t and future offset k in delta_indices, computes:
        relative_action[k] = action[t + k] - state[t]
    Then computes stats (mean, std, min, max, q01, q99) over all relative action chunks.

    Only joint-space (NON_EEF) actions are supported (simple subtraction).

    Args:
        dataset_root: Path to the dataset root directory.
        relative_action_keys: Action keys to compute relative stats for (e.g., ["left_arm", "right_arm"]).
            If None, defaults to ["left_arm", "right_arm"].
        delta_indices: Action horizon indices. If None, defaults to list(range(0, 50)).

    Returns:
        Dict mapping action key names to dicts of stat lists, ready for JSON serialization.
        E.g., {"left_arm": {"mean": [...], "std": [...], ...}, "right_arm": {...}}
    """
    import json

    dataset_root = Path(dataset_root)

    if relative_action_keys is None:
        relative_action_keys = ["left_arm", "right_arm"]
    if delta_indices is None:
        delta_indices = list(range(0, 50))

    # Load modality.json to get start/end indices for slicing
    with open(dataset_root / "meta" / "modality.json", "r") as f:
        modality_meta = json.load(f)

    # Load episodes metadata to get episode boundaries
    episodes = []
    with open(dataset_root / "meta" / "episodes.jsonl", "r") as f:
        for line in f:
            episodes.append(json.loads(line))

    # Load info for chunk size and data path pattern
    with open(dataset_root / "meta" / "info.json", "r") as f:
        info = json.load(f)
    chunk_size = info["chunks_size"]
    data_path_pattern = info["data_path"]

    rel_stats = {}
    for action_key in relative_action_keys:
        if action_key not in modality_meta.get("action", {}):
            print(f"Warning: action key '{action_key}' not found in modality.json, skipping")
            continue
        if action_key not in modality_meta.get("state", {}):
            print(f"Warning: state key '{action_key}' not found in modality.json, skipping")
            continue

        action_info = modality_meta["action"][action_key]
        state_info = modality_meta["state"][action_key]
        a_start, a_end = action_info["start"], action_info["end"]
        s_start, s_end = state_info["start"], state_info["end"]
        action_col = action_info.get("original_key", "action")
        state_col = state_info.get("original_key", "observation.state")

        max_offset = max(delta_indices)
        delta_arr = np.array(delta_indices)

        all_relative = []
        for ep in tqdm.tqdm(episodes, desc=f"Relative stats for {action_key}"):
            ep_idx = ep["episode_index"]
            chunk_idx = ep_idx // chunk_size
            parquet_path = dataset_root / data_path_pattern.format(
                episode_chunk=chunk_idx, episode_index=ep_idx
            )
            df = pd.read_parquet(parquet_path)

            # Extract action and state arrays for this key
            action_data = np.vstack(
                [np.asarray(x, dtype=np.float32)[a_start:a_end] for x in df[action_col]]
            )
            state_data = np.vstack(
                [np.asarray(x, dtype=np.float32)[s_start:s_end] for x in df[state_col]]
            )

            usable_length = len(df) - max_offset
            for t in range(usable_length):
                ref_state = state_data[t]  # shape: (dim,)
                action_chunk = action_data[t + delta_arr]  # shape: (horizon, dim)
                relative_chunk = action_chunk - ref_state  # broadcast subtract
                all_relative.append(relative_chunk)

        if not all_relative:
            print(f"Warning: no data for {action_key}, skipping")
            continue

        stacked = np.stack(all_relative, axis=0)  # (N, horizon, dim)
        # Compute stats along axis=0 → shape (horizon, dim), matching gr00t behavior.
        # Each horizon step gets its own statistics.
        rel_stats[action_key] = {
            "mean": np.mean(stacked, axis=0).astype(np.float32).tolist(),
            "std": np.std(stacked, axis=0).astype(np.float32).tolist(),
            "min": np.min(stacked, axis=0).astype(np.float32).tolist(),
            "max": np.max(stacked, axis=0).astype(np.float32).tolist(),
            "q01": np.quantile(stacked, 0.01, axis=0).astype(np.float32).tolist(),
            "q99": np.quantile(stacked, 0.99, axis=0).astype(np.float32).tolist(),
        }

    return rel_stats


def compute_stats(dataset, batch_size=8, num_workers=8, max_num_samples=None):
    """Compute mean/std and min/max statistics of all data keys in a LeRobotDataset."""
    if max_num_samples is None:
        max_num_samples = len(dataset)

    # for more info on why we need to set the same number of workers, see `load_from_videos`
    stats_patterns = get_stats_einops_patterns(dataset, num_workers)

    # mean and std will be computed incrementally while max and min will track the running value.
    mean, std, max, min = {}, {}, {}, {}
    for key in stats_patterns:
        mean[key] = torch.tensor(0.0).float()
        std[key] = torch.tensor(0.0).float()
        max[key] = torch.tensor(-float("inf")).float()
        min[key] = torch.tensor(float("inf")).float()

    def create_seeded_dataloader(dataset, batch_size, seed):
        generator = torch.Generator()
        generator.manual_seed(seed)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            generator=generator,
        )
        return dataloader

    # Note: Due to be refactored soon. The point of storing `first_batch` is to make sure we don't get
    # surprises when rerunning the sampler.
    first_batch = None
    running_item_count = 0  # for online mean computation
    dataloader = create_seeded_dataloader(dataset, batch_size, seed=1337)
    for i, batch in enumerate(
        tqdm.tqdm(dataloader, total=ceil(max_num_samples / batch_size), desc="Compute mean, min, max")
    ):
        this_batch_size = len(batch["index"])
        running_item_count += this_batch_size
        if first_batch is None:
            first_batch = deepcopy(batch)
        for key, pattern in stats_patterns.items():
            batch[key] = batch[key].float()
            
            # Handle point cloud reshaping (create a copy to avoid modifying original)
            tensor_for_stats = batch[key]
            if "pointcloud" in key and pattern == "b n c -> c":
                # Reshape flattened point cloud to (batch, n_points, 3) for proper coordinate-wise stats
                batch_size = batch[key].shape[0]
                n_coords = batch[key].shape[1]  # This is N*3 for flattened point clouds
                n_points = n_coords // 3
                tensor_for_stats = batch[key].view(batch_size, n_points, 3)
            
            # Numerically stable update step for mean computation.
            batch_mean = einops.reduce(tensor_for_stats, pattern, "mean")
            # Hint: to update the mean we need x̄ₙ = (Nₙ₋₁x̄ₙ₋₁ + Bₙxₙ) / Nₙ, where the subscript represents
            # the update step, N is the running item count, B is this batch size, x̄ is the running mean,
            # and x is the current batch mean. Some rearrangement is then required to avoid risking
            # numerical overflow. Another hint: Nₙ₋₁ = Nₙ - Bₙ. Rearrangement yields
            # x̄ₙ = x̄ₙ₋₁ + Bₙ * (xₙ - x̄ₙ₋₁) / Nₙ
            mean[key] = mean[key] + this_batch_size * (batch_mean - mean[key]) / running_item_count
            
            # Compute max/min using the same reshaped tensor
            max[key] = torch.maximum(max[key], einops.reduce(tensor_for_stats, pattern, "max"))
            min[key] = torch.minimum(min[key], einops.reduce(tensor_for_stats, pattern, "min"))

        if i == ceil(max_num_samples / batch_size) - 1:
            break

    first_batch_ = None
    running_item_count = 0  # for online std computation
    dataloader = create_seeded_dataloader(dataset, batch_size, seed=1337)
    for i, batch in enumerate(
        tqdm.tqdm(dataloader, total=ceil(max_num_samples / batch_size), desc="Compute std")
    ):
        this_batch_size = len(batch["index"])
        running_item_count += this_batch_size
        # Sanity check to make sure the batches are still in the same order as before.
        if first_batch_ is None:
            first_batch_ = deepcopy(batch)
            # Note: This assertion can fail due to different batch sizes between the two loops
            # when using shuffle=True with drop_last=False. This is not critical for statistics computation.
            # for key in stats_patterns:
            #     assert torch.equal(first_batch_[key], first_batch[key])
        for key, pattern in stats_patterns.items():
            batch[key] = batch[key].float()
            
            # Handle point cloud reshaping for std computation (create a copy to avoid modifying original)
            tensor_for_stats = batch[key]
            if "pointcloud" in key and pattern == "b n c -> c":
                # Reshape flattened point cloud to (batch, n_points, 3) for proper coordinate-wise stats
                batch_size = batch[key].shape[0]
                n_coords = batch[key].shape[1]  # This is N*3 for flattened point clouds
                n_points = n_coords // 3
                tensor_for_stats = batch[key].view(batch_size, n_points, 3)
            
            # Numerically stable update step for mean computation (where the mean is over squared
            # residuals).See notes in the mean computation loop above.
            batch_std = einops.reduce((tensor_for_stats - mean[key]) ** 2, pattern, "mean")
            std[key] = std[key] + this_batch_size * (batch_std - std[key]) / running_item_count

        if i == ceil(max_num_samples / batch_size) - 1:
            break

    for key in stats_patterns:
        std[key] = torch.sqrt(std[key])

    stats = {}
    for key in stats_patterns:
        stats[key] = {
            "mean": mean[key],
            "std": std[key],
            "max": max[key],
            "min": min[key],
        }
    return stats


def aggregate_stats(ls_datasets) -> dict[str, torch.Tensor]:
    """Aggregate stats of multiple LeRobot datasets into one set of stats without recomputing from scratch.

    The final stats will have the union of all data keys from each of the datasets.

    The final stats will have the union of all data keys from each of the datasets. For instance:
    - new_max = max(max_dataset_0, max_dataset_1, ...)
    - new_min = min(min_dataset_0, min_dataset_1, ...)
    - new_mean = (mean of all data)
    - new_std = (std of all data)
    """
    data_keys = set()
    for dataset in ls_datasets:
        data_keys.update(dataset.meta.stats.keys())
    stats = {k: {} for k in data_keys}
    for data_key in data_keys:
        for stat_key in ["min", "max"]:
            # compute `max(dataset_0["max"], dataset_1["max"], ...)`
            stats[data_key][stat_key] = einops.reduce(
                torch.stack(
                    [ds.meta.stats[data_key][stat_key] for ds in ls_datasets if data_key in ds.meta.stats],
                    dim=0,
                ),
                "n ... -> ...",
                stat_key,
            )
        total_samples = sum(d.num_frames for d in ls_datasets if data_key in d.meta.stats)
        # Compute the "sum" statistic by multiplying each mean by the number of samples in the respective
        # dataset, then divide by total_samples to get the overall "mean".
        # NOTE: the brackets around (d.num_frames / total_samples) are needed tor minimize the risk of
        # numerical overflow!
        stats[data_key]["mean"] = sum(
            d.meta.stats[data_key]["mean"] * (d.num_frames / total_samples)
            for d in ls_datasets
            if data_key in d.meta.stats
        )
        # The derivation for standard deviation is a little more involved but is much in the same spirit as
        # the computation of the mean.
        # Given two sets of data where the statistics are known:
        # σ_combined = sqrt[ (n1 * (σ1^2 + d1^2) + n2 * (σ2^2 + d2^2)) / (n1 + n2) ]
        # where d1 = μ1 - μ_combined, d2 = μ2 - μ_combined
        # NOTE: the brackets around (d.num_frames / total_samples) are needed tor minimize the risk of
        # numerical overflow!
        stats[data_key]["std"] = torch.sqrt(
            sum(
                (
                    d.meta.stats[data_key]["std"] ** 2
                    + (d.meta.stats[data_key]["mean"] - stats[data_key]["mean"]) ** 2
                )
                * (d.num_frames / total_samples)
                for d in ls_datasets
                if data_key in d.meta.stats
            )
        )
    return stats
