import contextlib
import logging
import random
import json
import copy
import os
import shutil
import cv2
import polars as pl
from pathlib import Path
from typing import Callable, Dict
from datetime import datetime

import datasets
import numpy as np
import packaging.version
import PIL.Image as Image
import torch
import torch.utils

from torch.utils.data import ConcatDataset, Subset
from torch.utils.data.dataloader import default_collate
import torch.nn.functional as F
from torchvision import transforms as T
import pickle

import transformers
from transformers.trainer_pt_utils import LabelSmoother
from torchvision.transforms import v2

IGNORE_TOKEN_ID = LabelSmoother.ignore_index

from datasets import concatenate_datasets, load_dataset, Dataset

from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.constants import REPOCARD_NAME
from huggingface_hub.errors import RevisionNotFoundError

from cosmos_policy.datasets.lerobot.utils import (
    DEFAULT_FEATURES,
    DEFAULT_IMAGE_PATH,
    INFO_PATH,
    TASKS_PATH,
    append_jsonlines,
    check_delta_timestamps,
    check_timestamps_sync,
    check_version_compatibility,
    create_empty_dataset_info,
    create_lerobot_dataset_card,
    embed_images,
    get_delta_indices,
    get_episode_data_index,
    get_features_from_robot,
    get_hf_features_from_features,
    get_safe_version,
    hf_transform_to_torch,
    is_valid_version,
    load_episodes,
    load_episodes_stats,
    load_info,
    load_stats,
    load_tasks,
    validate_episode_buffer,
    validate_frame,
    write_episode,
    write_episode_stats,
    write_info,
    write_json,
)
from cosmos_policy.datasets.lerobot.constants import HF_HOME, HF_LEROBOT_HOME, OBS_ROBOT
from cosmos_policy.datasets.lerobot.video_utils import (
    VideoFrame,
    encode_video_frames,
    decode_video_frames,
    get_video_info,
)
from cosmos_policy.datasets.lerobot.image_writer import AsyncImageWriter, write_image
from cosmos_policy.datasets.lerobot.compute_stats import compute_episode_stats, aggregate_stats
from cosmos_policy.datasets.lerobot.data_utils import preprocess_image
from cosmos_policy.datasets.lerobot.oxe_configs import OXE_DATASET_CONFIGS
from cosmos_policy.datasets.lerobot.mixtures import OXE_NAMED_MIXTURES
import hashlib
from tabulate import tabulate

CODEBASE_VERSION = "v2.1"

class IndexedDataset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[int(self.indices[idx])]


def allocate_samples_by_weights(weights, total_size):
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / weights.sum()

    raw_counts = weights * total_size
    counts = np.floor(raw_counts).astype(int)

    return counts

def tensor_to_list(obj):
    """
    递归地将包含 torch.Tensor 的结构转换为纯 Python 数据结构。
    """
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: tensor_to_list(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [tensor_to_list(v) for v in obj]
    else:
        return obj

def save_to_json(data, path):
    """
    保存数据为 JSON 文件。
    """
    converted_data = tensor_to_list(data)
    
    # 创建路径中不存在的文件夹
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # 保存为 JSON 文件
    with open(path, 'w') as f:
        json.dump(converted_data, f, indent=4)

def resolve_delta_timestamps(ds_meta, chunk_size, use_reward=False):

    delta_timestamps = {}

    obs_indices = list(range(chunk_size))
    act_indices = list(range(chunk_size))
    rew_indices = list(range(chunk_size)) if use_reward else None

    for key in ds_meta.features:

        if key == "next.reward" and rew_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in rew_indices]

        elif key == "action":
            delta_timestamps[key] = [i / ds_meta.fps for i in act_indices]

        elif key.startswith("observation."):
            delta_timestamps[key] = [i / ds_meta.fps for i in obs_indices]

    return delta_timestamps or None

def duplicate_array(arr, total_num_copies):
    """
    Duplicates a NumPy array multiple times along a new first axis.

    Args:
        arr (numpy.ndarray): The input array to duplicate
        total_num_copies (int): Total number of copies to have in the end

    Returns:
        numpy.ndarray: A new array with shape (total_num_copies, *arr.shape)
    """
    # Create a new array by stacking the original array multiple times
    return np.stack([arr] * total_num_copies)

def safe_hash(input_tuple):
    # keep 128 bits of the hash
    tuple_string = repr(input_tuple).encode("utf-8")
    sha256 = hashlib.sha256()
    sha256.update(tuple_string)

    seed = int(sha256.hexdigest(), 16)

    return seed & 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF

class LeRobotDatasetMetadata:
    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        revision: str | None = None,
        force_cache_sync: bool = False,
    ):
        self.repo_id = repo_id
        self.revision = revision if revision else CODEBASE_VERSION
        self.root = Path(root) if root is not None else HF_LEROBOT_HOME / repo_id

        try:
            if force_cache_sync:
                raise FileNotFoundError
            self.load_metadata()
        except (FileNotFoundError, NotADirectoryError):
            if is_valid_version(self.revision):
                self.revision = get_safe_version(self.repo_id, self.revision)

            (self.root / "meta").mkdir(exist_ok=True, parents=True)
            self.pull_from_repo(allow_patterns="meta/")
            self.load_metadata()
            
    def restrict_image_features(self, features: dict[str, dict], max_feature=8) -> dict[str, dict]:
        """Restricts the number of image features to a maximum number."""
        image_features = {k: v for k, v in features.items() if v["dtype"] in ["image", "video"]}
        if len(image_features) > max_feature:
            logging.warning(
                f"Found {len(image_features)} image features, restricting to {max_feature}."
            )
            num_features = len(image_features)
            image_features = dict(list(image_features.items())[:num_features-max_feature])
        # remove feature not in image features
        feature_to_return = features.copy()
        if len(image_features) > max_feature:
            for k in features.keys():
                if k in image_features.keys():
                    feature_to_return.pop(k)
        return feature_to_return
    def load_metadata(self):
        self.info = load_info(self.root)
        self.info['features'] = self.restrict_image_features(self.info['features'])
        check_version_compatibility(self.repo_id, self._version, CODEBASE_VERSION)
        self.tasks, self.task_to_task_index = load_tasks(self.root)
        self.episodes = load_episodes(self.root)
        self.stats = load_stats(self.root)
        if self.stats == None:
            self.episodes_stats = load_episodes_stats(self.root)
            self.stats = aggregate_stats(list(self.episodes_stats.values()))
        # if self._version < packaging.version.parse("v2.1"):
        #     self.stats = load_stats(self.root)
        #     self.episodes_stats = backward_compatible_episodes_stats(self.stats, self.episodes)
        # else:
        #     self.episodes_stats = load_episodes_stats(self.root)
        #     self.stats = aggregate_stats(list(self.episodes_stats.values()))

    def pull_from_repo(
        self,
        allow_patterns: list[str] | str | None = None,
        ignore_patterns: list[str] | str | None = None,
    ) -> None:
        snapshot_download(
            self.repo_id,
            repo_type="dataset",
            revision=self.revision,
            local_dir=self.root,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
        )

    @property
    def _version(self) -> packaging.version.Version:
        """Codebase version used to create this dataset."""
        return packaging.version.parse(self.info["codebase_version"])

    def get_data_file_path(self, ep_index: int) -> Path:
        ep_chunk = self.get_episode_chunk(ep_index)
        fpath = self.data_path.format(episode_chunk=ep_chunk, episode_index=ep_index)
        return Path(fpath)

    def get_video_file_path(self, ep_index: int, vid_key: str) -> Path:
        ep_chunk = self.get_episode_chunk(ep_index)
        fpath = self.video_path.format(episode_chunk=ep_chunk, video_key=vid_key, episode_index=ep_index)
        return Path(fpath)

    def get_episode_chunk(self, ep_index: int) -> int:
        return ep_index // self.chunks_size

    @property
    def data_path(self) -> str:
        """Formattable string for the parquet files."""
        return self.info["data_path"]

    @property
    def video_path(self) -> str | None:
        """Formattable string for the video files."""
        return self.info["video_path"]

    @property
    def robot_type(self) -> str | None:
        """Robot type used in recording this dataset."""
        return self.info["robot_type"]

    @property
    def fps(self) -> int:
        """Frames per second used during data collection."""
        return self.info["fps"]

    @property
    def features(self) -> dict[str, dict]:
        """All features contained in the dataset."""
        return self.info["features"]

    @property
    def image_keys(self) -> list[str]:
        """Keys to access visual modalities stored as images."""
        return [key for key, ft in self.features.items() if ft["dtype"] == "image"]

    @property
    def video_keys(self) -> list[str]:
        """Keys to access visual modalities stored as videos."""
        return [key for key, ft in self.features.items() if ft["dtype"] == "video"]

    @property
    def camera_keys(self) -> list[str]:
        """Keys to access visual modalities (regardless of their storage method)."""
        return [key for key, ft in self.features.items() if ft["dtype"] in ["video", "image"]]

    @property
    def names(self) -> dict[str, list | dict]:
        """Names of the various dimensions of vector modalities."""
        return {key: ft["names"] for key, ft in self.features.items()}

    @property
    def shapes(self) -> dict:
        """Shapes for the different features."""
        return {key: tuple(ft["shape"]) for key, ft in self.features.items()}

    @property
    def total_episodes(self) -> int:
        """Total number of episodes available."""
        return self.info["total_episodes"]

    @property
    def total_frames(self) -> int:
        """Total number of frames saved in this dataset."""
        return self.info["total_frames"]

    @property
    def total_tasks(self) -> int:
        """Total number of different tasks performed in this dataset."""
        return self.info["total_tasks"]

    @property
    def total_chunks(self) -> int:
        """Total number of chunks (groups of episodes)."""
        return self.info["total_chunks"]

    @property
    def chunks_size(self) -> int:
        """Max number of episodes per chunk."""
        return self.info["chunks_size"]

    def get_task_index(self, task: str) -> int | None:
        """
        Given a task in natural language, returns its task_index if the task already exists in the dataset,
        otherwise return None.
        """
        return self.task_to_task_index.get(task, None)

    def add_task(self, task: str):
        """
        Given a task in natural language, add it to the dictionary of tasks.
        """
        if task in self.task_to_task_index:
            raise ValueError(f"The task '{task}' already exists and can't be added twice.")

        task_index = self.info["total_tasks"]
        self.task_to_task_index[task] = task_index
        self.tasks[task_index] = task
        self.info["total_tasks"] += 1

        task_dict = {
            "task_index": task_index,
            "task": task,
        }
        append_jsonlines(task_dict, self.root / TASKS_PATH)

    def save_episode(
        self,
        episode_index: int,
        episode_length: int,
        episode_tasks: list[str],
        episode_stats: dict[str, dict],
    ) -> None:
        self.info["total_episodes"] += 1
        self.info["total_frames"] += episode_length

        chunk = self.get_episode_chunk(episode_index)
        if chunk >= self.total_chunks:
            self.info["total_chunks"] += 1

        self.info["splits"] = {"train": f"0:{self.info['total_episodes']}"}
        self.info["total_videos"] += len(self.video_keys)
        if len(self.video_keys) > 0:
            self.update_video_info()

        write_info(self.info, self.root)

        episode_dict = {
            "episode_index": episode_index,
            "tasks": episode_tasks,
            "length": episode_length,
        }
        self.episodes[episode_index] = episode_dict
        write_episode(episode_dict, self.root)

        self.episodes_stats[episode_index] = episode_stats
        self.stats = aggregate_stats([self.stats, episode_stats]) if self.stats else episode_stats
        write_episode_stats(episode_index, episode_stats, self.root)

    def update_video_info(self) -> None:
        """
        Warning: this function writes info from first episode videos, implicitly assuming that all videos have
        been encoded the same way. Also, this means it assumes the first episode exists.
        """
        for key in self.video_keys:
            if not self.features[key].get("info", None):
                video_path = self.root / self.get_video_file_path(ep_index=0, vid_key=key)
                self.info["features"][key]["info"] = get_video_info(video_path)

    def __repr__(self):
        feature_keys = list(self.features)
        return (
            f"{self.__class__.__name__}({{\n"
            f"    Repository ID: '{self.repo_id}',\n"
            f"    Total episodes: '{self.total_episodes}',\n"
            f"    Total frames: '{self.total_frames}',\n"
            f"    Features: '{feature_keys}',\n"
            "})',\n"
        )

    @classmethod
    def create(
        cls,
        repo_id: str,
        fps: int,
        root: str | Path | None = None,
        robot: None = None,
        robot_type: str | None = None,
        features: dict | None = None,
        use_videos: bool = True,
    ) -> "LeRobotDatasetMetadata":
        """Creates metadata for a LeRobotDataset."""
        obj = cls.__new__(cls)
        obj.repo_id = repo_id
        obj.root = Path(root) if root is not None else HF_LEROBOT_HOME / repo_id

        obj.root.mkdir(parents=True, exist_ok=False)

        if robot is not None:
            features = get_features_from_robot(robot, use_videos)
            robot_type = robot.robot_type
            if not all(cam.fps == fps for cam in robot.cameras.values()):
                logging.warning(
                    f"Some cameras in your {robot.robot_type} robot don't have an fps matching the fps of your dataset."
                    "In this case, frames from lower fps cameras will be repeated to fill in the blanks."
                )
        elif features is None:
            raise ValueError(
                "Dataset features must either come from a Robot or explicitly passed upon creation."
            )
        else:
            # TODO(aliberts, rcadene): implement sanity check for features
            features = {**features, **DEFAULT_FEATURES}

            # check if none of the features contains a "/" in their names,
            # as this would break the dict flattening in the stats computation, which uses '/' as separator
            for key in features:
                if "/" in key:
                    raise ValueError(f"Feature names should not contain '/'. Found '/' in feature '{key}'.")

            features = {**features, **DEFAULT_FEATURES}

        obj.tasks, obj.task_to_task_index = {}, {}
        obj.episodes_stats, obj.stats, obj.episodes = {}, {}, {}
        obj.info = create_empty_dataset_info(CODEBASE_VERSION, fps, robot_type, features, use_videos)
        if len(obj.video_keys) > 0 and not use_videos:
            raise ValueError()
        write_json(obj.info, obj.root / INFO_PATH)
        obj.revision = None
        return obj
    
    @classmethod
    def create_with_stats_feats(
        cls, 
        stats, 
        features,
        fps = 30,
        robot_type = "all",
        use_videos = True,
        ) -> "LeRobotDatasetMetadata":
        obj = cls.__new__(cls)
        obj.stats = stats
        obj.info = create_empty_dataset_info(CODEBASE_VERSION, fps, robot_type, features, use_videos)
        return obj


class LeRobotDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        episodes: list[int] | None = None,
        image_transforms: Callable | None = None,
        wrist_image_transforms: Callable | None = None,
        delta_timestamps: dict[list[float]] | None = None,
        tolerance_s: float = 1e-4,
        revision: str | None = None,
        force_cache_sync: bool = False,
        download_videos: bool = True,
        video_backend: str | None = None,
        keep_img_keys: str | None = None,
        dataset_name: str = "default",
    ):
        """
        2 modes are available for instantiating this class, depending on 2 different use cases:

        1. Your dataset already exists:
            - On your local disk in the 'root' folder. This is typically the case when you recorded your
              dataset locally and you may or may not have pushed it to the hub yet. Instantiating this class
              with 'root' will load your dataset directly from disk. This can happen while you're offline (no
              internet connection).

            - On the Hugging Face Hub at the address https://huggingface.co/datasets/{repo_id} and not on
              your local disk in the 'root' folder. Instantiating this class with this 'repo_id' will download
              the dataset from that address and load it, pending your dataset is compliant with
              codebase_version v2.0. If your dataset has been created before this new format, you will be
              prompted to convert it using our conversion script from v1.6 to v2.0, which you can find at
              lerobot/common/datasets/v2/convert_dataset_v1_to_v2.py.


        2. Your dataset doesn't already exists (either on local disk or on the Hub): you can create an empty
           LeRobotDataset with the 'create' classmethod. This can be used for recording a dataset or port an
           existing dataset to the LeRobotDataset format.


        In terms of files, LeRobotDataset encapsulates 3 main things:
            - metadata:
                - info contains various information about the dataset like shapes, keys, fps etc.
                - stats stores the dataset statistics of the different modalities for normalization
                - tasks contains the prompts for each task of the dataset, which can be used for
                  task-conditioned training.
            - hf_dataset (from datasets.Dataset), which will read any values from parquet files.
            - videos (optional) from which frames are loaded to be synchronous with data from parquet files.

        A typical LeRobotDataset looks like this from its root path:
        .
        ├── data
        │   ├── chunk-000
        │   │   ├── episode_000000.parquet
        │   │   ├── episode_000001.parquet
        │   │   ├── episode_000002.parquet
        │   │   └── ...
        │   ├── chunk-001
        │   │   ├── episode_001000.parquet
        │   │   ├── episode_001001.parquet
        │   │   ├── episode_001002.parquet
        │   │   └── ...
        │   └── ...
        ├── meta
        │   ├── episodes.jsonl
        │   ├── info.json
        │   ├── stats.json
        │   └── tasks.jsonl
        └── videos
            ├── chunk-000
            │   ├── observation.images.laptop
            │   │   ├── episode_000000.mp4
            │   │   ├── episode_000001.mp4
            │   │   ├── episode_000002.mp4
            │   │   └── ...
            │   ├── observation.images.phone
            │   │   ├── episode_000000.mp4
            │   │   ├── episode_000001.mp4
            │   │   ├── episode_000002.mp4
            │   │   └── ...
            ├── chunk-001
            └── ...

        Note that this file-based structure is designed to be as versatile as possible. The files are split by
        episodes which allows a more granular control over which episodes one wants to use and download. The
        structure of the dataset is entirely described in the info.json file, which can be easily downloaded
        or viewed directly on the hub before downloading any actual data. The type of files used are very
        simple and do not need complex tools to be read, it only uses .parquet, .json and .mp4 files (and .md
        for the README).

        Args:
            repo_id (str): This is the repo id that will be used to fetch the dataset. Locally, the dataset
                will be stored under root/repo_id.
            root (Path | None, optional): Local directory to use for downloading/writing files. You can also
                set the LEROBOT_HOME environment variable to point to a different location. Defaults to
                '~/.cache/huggingface/lerobot'.
            episodes (list[int] | None, optional): If specified, this will only load episodes specified by
                their episode_index in this list. Defaults to None.
            image_transforms (Callable | None, optional): You can pass standard v2 image transforms from
                torchvision.transforms.v2 here which will be applied to visual modalities (whether they come
                from videos or images). Defaults to None.
            delta_timestamps (dict[list[float]] | None, optional): _description_. Defaults to None.
            tolerance_s (float, optional): Tolerance in seconds used to ensure data timestamps are actually in
                sync with the fps value. It is used at the init of the dataset to make sure that each
                timestamps is separated to the next by 1/fps +/- tolerance_s. This also applies to frames
                decoded from video files. It is also used to check that `delta_timestamps` (when provided) are
                multiples of 1/fps. Defaults to 1e-4.
            revision (str, optional): An optional Git revision id which can be a branch name, a tag, or a
                commit hash. Defaults to current codebase version tag.
            sync_cache_first (bool, optional): Flag to sync and refresh local files first. If True and files
                are already present in the local cache, this will be faster. However, files loaded might not
                be in sync with the version on the hub, especially if you specified 'revision'. Defaults to
                False.
            download_videos (bool, optional): Flag to download the videos. Note that when set to True but the
                video files are already present on local disk, they won't be downloaded again. Defaults to
                True.
            video_backend (str | None, optional): Video backend to use for decoding videos. There is currently
                a single option which is the pyav decoder used by Torchvision. Defaults to pyav.
        """
        super().__init__()
        # print("__init__ 方法被调用")
        self.repo_id = repo_id
        self.root = Path(root) if root else HF_LEROBOT_HOME / repo_id
        self.image_transforms = image_transforms
        self.wrist_image_transforms = wrist_image_transforms
        # print(self.image_transforms, self.wrist_image_transforms)
        self.delta_timestamps = delta_timestamps
        self.episodes = episodes
        self.tolerance_s = tolerance_s
        self.revision = revision if revision else CODEBASE_VERSION
        self.video_backend = video_backend if video_backend else "pyav"
        self.delta_indices = None
        self.keep_img_keys = keep_img_keys
        self.dataset_name = dataset_name

        # Unused attributes
        self.image_writer = None
        self.episode_buffer = None

        self.root.mkdir(exist_ok=True, parents=True)

        # Load metadata
        self.meta = LeRobotDatasetMetadata(
            self.repo_id, self.root, self.revision, force_cache_sync=force_cache_sync
        )
        # print(f"Episodes in the dataset: {episodes}")
        if self.episodes is not None and self.meta._version >= packaging.version.parse("v2.1"):
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] - Loading episodes stats...")
            episodes_stats = [self.meta.episodes_stats[ep_idx] for ep_idx in self.episodes]
            self.stats = aggregate_stats(episodes_stats)

        # Load actual data
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] - Trying to load dataset {self.repo_id}...")
        try:
            if force_cache_sync:
                raise FileNotFoundError
            # assert all((self.root / fpath).is_file() for fpath in self.get_episodes_file_paths())
            self.hf_dataset = self.load_hf_dataset()
        except (AssertionError, FileNotFoundError, NotADirectoryError):
            self.revision = get_safe_version(self.repo_id, self.revision)
            self.download_episodes(download_videos)
            self.hf_dataset = self.load_hf_dataset()
            
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] - Dataset loaded successfully, loading timestamps.")

        self.episode_data_index = get_episode_data_index(self.meta.episodes, self.episodes)

        # Check timestamps
        timestamps = torch.stack(list(self.hf_dataset["timestamp"])).numpy()
        episode_indices = torch.stack(list(self.hf_dataset["episode_index"])).numpy()
        ep_data_index_np = {k: t.numpy() for k, t in self.episode_data_index.items()}
        
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] - Checking timestamps sync status...")
        
        check_timestamps_sync(timestamps, episode_indices, ep_data_index_np, self.fps, self.tolerance_s)

        # Setup delta_indices
        if self.delta_timestamps is not None:
            check_delta_timestamps(self.delta_timestamps, self.fps, self.tolerance_s)
            self.delta_indices = get_delta_indices(self.delta_timestamps, self.fps)

    def push_to_hub(
        self,
        branch: str | None = None,
        tags: list | None = None,
        license: str | None = "apache-2.0",
        tag_version: bool = True,
        push_videos: bool = True,
        private: bool = False,
        allow_patterns: list[str] | str | None = None,
        upload_large_folder: bool = False,
        **card_kwargs,
    ) -> None:
        ignore_patterns = ["images/"]
        if not push_videos:
            ignore_patterns.append("videos/")

        hub_api = HfApi()
        hub_api.create_repo(
            repo_id=self.repo_id,
            private=private,
            repo_type="dataset",
            exist_ok=True,
        )
        if branch:
            hub_api.create_branch(
                repo_id=self.repo_id,
                branch=branch,
                revision=self.revision,
                repo_type="dataset",
                exist_ok=True,
            )

        upload_kwargs = {
            "repo_id": self.repo_id,
            "folder_path": self.root,
            "repo_type": "dataset",
            "revision": branch,
            "allow_patterns": allow_patterns,
            "ignore_patterns": ignore_patterns,
        }
        if upload_large_folder:
            hub_api.upload_large_folder(**upload_kwargs)
        else:
            hub_api.upload_folder(**upload_kwargs)

        if not hub_api.file_exists(self.repo_id, REPOCARD_NAME, repo_type="dataset", revision=branch):
            card = create_lerobot_dataset_card(
                tags=tags, dataset_info=self.meta.info, license=license, **card_kwargs
            )
            card.push_to_hub(repo_id=self.repo_id, repo_type="dataset", revision=branch)

        if tag_version:
            with contextlib.suppress(RevisionNotFoundError):
                hub_api.delete_tag(self.repo_id, tag=CODEBASE_VERSION, repo_type="dataset")
            hub_api.create_tag(self.repo_id, tag=CODEBASE_VERSION, revision=branch, repo_type="dataset")

    def pull_from_repo(
        self,
        allow_patterns: list[str] | str | None = None,
        ignore_patterns: list[str] | str | None = None,
    ) -> None:
        snapshot_download(
            self.repo_id,
            repo_type="dataset",
            revision=self.revision,
            local_dir=self.root,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
        )

    def download_episodes(self, download_videos: bool = True) -> None:
        """Downloads the dataset from the given 'repo_id' at the provided version. If 'episodes' is given, this
        will only download those episodes (selected by their episode_index). If 'episodes' is None, the whole
        dataset will be downloaded. Thanks to the behavior of snapshot_download, if the files are already present
        in 'local_dir', they won't be downloaded again.
        """
        # TODO(rcadene, aliberts): implement faster transfer
        # https://huggingface.co/docs/huggingface_hub/en/guides/download#faster-downloads
        files = None
        ignore_patterns = None if download_videos else "videos/"
        if self.episodes is not None:
            files = self.get_episodes_file_paths()

        self.pull_from_repo(allow_patterns=files, ignore_patterns=ignore_patterns)

    def get_episodes_file_paths(self) -> list[Path]:
        episodes = self.episodes if self.episodes is not None else list(range(self.meta.total_episodes))
        fpaths = [str(self.meta.get_data_file_path(ep_idx)) for ep_idx in episodes]
        if len(self.meta.video_keys) > 0:
            video_files = [
                str(self.meta.get_video_file_path(ep_idx, vid_key))
                for vid_key in self.meta.video_keys
                for ep_idx in episodes
            ]
            fpaths += video_files

        return fpaths

    def load_hf_dataset(self) -> datasets.Dataset:
        """hf_dataset contains all the observations, states, actions, rewards, etc."""
        if self.episodes is None:
            # path = str(self.root / "data")
            path = str(self.root / "merged.parquet")
            # hf_dataset = parquet_to_dataset(parquet_file=path, split="train")
            hf_dataset = load_dataset("parquet", data_files=path, split="train")
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] - Dataset length is {len(hf_dataset)}")
            # hf_dataset = load_dataset("parquet", data_dir=path, split="train")
        else:
            files = [str(self.root / self.meta.get_data_file_path(ep_idx)) for ep_idx in self.episodes]
            hf_dataset = load_dataset("parquet", data_files=files, split="train")

        # TODO(aliberts): hf_dataset.set_format("torch")
        hf_dataset.set_transform(hf_transform_to_torch)
        # hf_dataset.set_format("torch")
        return hf_dataset

    def create_hf_dataset(self) -> datasets.Dataset:
        features = get_hf_features_from_features(self.features)
        ft_dict = {col: [] for col in features}
        hf_dataset = datasets.Dataset.from_dict(ft_dict, features=features, split="train")

        # TODO(aliberts): hf_dataset.set_format("torch")
        hf_dataset.set_transform(hf_transform_to_torch)
        return hf_dataset

    @property
    def fps(self) -> int:
        """Frames per second used during data collection."""
        return self.meta.fps

    @property
    def num_frames(self) -> int:
        """Number of frames in selected episodes."""
        return len(self.hf_dataset) if self.hf_dataset is not None else self.meta.total_frames

    @property
    def num_episodes(self) -> int:
        """Number of episodes selected."""
        return len(self.episodes) if self.episodes is not None else self.meta.total_episodes

    @property
    def features(self) -> dict[str, dict]:
        return self.meta.features

    @property
    def hf_features(self) -> datasets.Features:
        """Features of the hf_dataset."""
        if self.hf_dataset is not None:
            return self.hf_dataset.features
        else:
            return get_hf_features_from_features(self.features)

    def expand_true(self, mask, k=2):
        mask = mask.clone()
        true_idx = mask.nonzero(as_tuple=True)[0]
        if len(true_idx) > 0:
            start = true_idx[0].item()
            new_start = max(0, start - k)
            mask[new_start:] = True   # 注意这里是从 new_start 到最后都置为 True
        return mask

    def _get_query_indices(self, idx: int, ep_idx: int) -> tuple[dict[str, list[int | bool]]]:
        ep_start = self.episode_data_index["from"][ep_idx]
        ep_end = self.episode_data_index["to"][ep_idx]
        # delta_indices:{"action" : [1, 2, 3, 4, 5]}
        query_indices = {
            key: [max(ep_start.item(), min(ep_end.item() - 1, idx + delta)) for delta in delta_idx]
            for key, delta_idx in self.delta_indices.items()
        }
        # query_indices["observation.images.image"] = query_indices["action"]
        # print(query_indices)
        padding = {  # Pad values outside of current episode range
            f"{key}_is_pad": torch.BoolTensor(
                [(idx + delta < ep_start.item()) | (idx + delta >= ep_end.item()) for delta in delta_idx]
            )
            for key, delta_idx in self.delta_indices.items()
        }
        return query_indices, padding

    def _get_query_timestamps(
        self,
        current_ts: float,
        query_indices: dict[str, list[int]] | None = None,
    ) -> dict[str, list[float]]:
        query_timestamps = {}
        for key in self.meta.video_keys:
            if query_indices is not None and key in query_indices:
                timestamps = self.hf_dataset.select(query_indices[key])["timestamp"]
                query_timestamps[key] = torch.stack(timestamps).tolist()
            else:
                query_timestamps[key] = [current_ts]
        # for key, timestamps in query_timestamps.items():
        #     print(key, timestamps)
        return query_timestamps

    def _query_hf_dataset(self, query_indices: dict[str, list[int]]) -> dict:
        return {
            key: torch.stack(list(self.hf_dataset.select(q_idx)[key]))
            for key, q_idx in query_indices.items()
            if key not in self.meta.video_keys
        }

    def _query_videos(self, query_timestamps: dict[str, list[float]], ep_idx: int) -> dict[str, torch.Tensor]:
        """Note: When using data workers (e.g. DataLoader with num_workers>0), do not call this function
        in the main process (e.g. by using a second Dataloader with num_workers=0). It will result in a
        Segmentation Fault. This probably happens because a memory reference to the video loader is created in
        the main process and a subprocess fails to access it.
        """
        item = {}
        for vid_key, query_ts in query_timestamps.items():
            video_path = self.root / self.meta.get_video_file_path(ep_idx, vid_key)
            # frames = decode_video_frames_torchvision(
            #     video_path, query_ts, self.tolerance_s, self.video_backend
            # )
            frames = decode_video_frames(video_path, query_ts, self.tolerance_s, self.video_backend, 
                                         return_type="image", worker_count=10)
            # print(vid_key, frames.shape)
            item[vid_key] = frames

        return item

    def _add_padding_keys(self, item: dict, padding: dict[str, list[bool]]) -> dict:
        for key, val in padding.items():
            item[key] = torch.BoolTensor(val)
        return item

    def __len__(self):
        return self.num_frames
    
    def resize_with_pad(self, img, width, height, pad_value=-1):
        # assume no-op when width height fits already
        need_expand = False
        if img.ndim != 4:
            need_expand = True
            img = img.unsqueeze(1)
            # raise ValueError(f"(b,c,h,w) expected, but {img.shape}")

        cur_height, cur_width = img.shape[2:]

        ratio = max(cur_width / width, cur_height / height)
        resized_height = int(cur_height / ratio)
        resized_width = int(cur_width / ratio)
        resized_img = F.interpolate(
            img, size=(resized_height, resized_width), mode="bilinear", align_corners=False
        )

        pad_height = max(0, int(height - resized_height))
        pad_width = max(0, int(width - resized_width))

        # pad on left and top of image
        padded_img = F.pad(resized_img, (pad_width, 0, pad_height, 0), value=pad_value)
        if need_expand:
            padded_img = padded_img.squeeze(1)
        return padded_img
    
    
    def resize_numpy(self, img):
        return cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)

    def __getitem__(self, idx) -> dict:
        # print(f"Idx:{idx}")
        item = self.hf_dataset[idx]
        ep_idx = item["episode_index"].item()
        
        query_indices = None
        if self.delta_indices is not None:
            query_indices, padding = self._get_query_indices(idx, ep_idx)
            # print(query_indices)
            query_result = self._query_hf_dataset(query_indices) # read action and state
            item = {**item, **padding}
            for key, val in query_result.items():
                item[key] = val
            
        if len(self.meta.video_keys) > 0:
            current_ts = item["timestamp"].item()
            query_timestamps = self._get_query_timestamps(current_ts, query_indices)
            video_frames = self._query_videos(query_timestamps, ep_idx)
            item = {**video_frames, **item}
        
        if self.image_transforms is not None:
            image_keys = self.meta.camera_keys
            for cam in image_keys:
                # item[cam] = [self.resize_numpy(img) for img in item[cam]]
                item[cam] = self.image_transforms(item[cam])
                # print(item[cam][0].shape, self.image_transforms)
        # Add task as a string
        task_idx = item["task_index"].item()
        item["task"] = self.meta.tasks[task_idx]
        item["dataset_name"] = self.dataset_name

        return item

    def __repr__(self):
        feature_keys = list(self.features)
        return (
            f"{self.__class__.__name__}({{\n"
            f"    Repository ID: '{self.repo_id}',\n"
            f"    Number of selected episodes: '{self.num_episodes}',\n"
            f"    Number of selected samples: '{self.num_frames}',\n"
            f"    Features: '{feature_keys}',\n"
            "})',\n"
        )

    def create_episode_buffer(self, episode_index: int | None = None) -> dict:
        current_ep_idx = self.meta.total_episodes if episode_index is None else episode_index
        ep_buffer = {}
        # size and task are special cases that are not in self.features
        ep_buffer["size"] = 0
        ep_buffer["task"] = []
        for key in self.features:
            ep_buffer[key] = current_ep_idx if key == "episode_index" else []
        return ep_buffer

    def _get_image_file_path(self, episode_index: int, image_key: str, frame_index: int) -> Path:
        fpath = DEFAULT_IMAGE_PATH.format(
            image_key=image_key, episode_index=episode_index, frame_index=frame_index
        )
        return self.root / fpath

    def _save_image(self, image: torch.Tensor | np.ndarray | Image.Image, fpath: Path) -> None:
        if self.image_writer is None:
            if isinstance(image, torch.Tensor):
                image = image.cpu().numpy()
            write_image(image, fpath)
        else:
            self.image_writer.save_image(image=image, fpath=fpath)

    def add_frame(self, frame: dict) -> None:
        """
        This function only adds the frame to the episode_buffer. Apart from images — which are written in a
        temporary directory — nothing is written to disk. To save those frames, the 'save_episode()' method
        then needs to be called.
        """
        # Convert torch to numpy if needed
        for name in frame:
            if isinstance(frame[name], torch.Tensor):
                frame[name] = frame[name].numpy()

        validate_frame(frame, self.features)

        if self.episode_buffer is None:
            self.episode_buffer = self.create_episode_buffer()

        # Automatically add frame_index and timestamp to episode buffer
        frame_index = self.episode_buffer["size"]
        timestamp = frame.pop("timestamp") if "timestamp" in frame else frame_index / self.fps
        self.episode_buffer["frame_index"].append(frame_index)
        self.episode_buffer["timestamp"].append(timestamp)

        # Add frame features to episode_buffer
        for key in frame:
            if key == "task":
                # Note: we associate the task in natural language to its task index during `save_episode`
                self.episode_buffer["task"].append(frame["task"])
                continue

            if key not in self.features:
                raise ValueError(
                    f"An element of the frame is not in the features. '{key}' not in '{self.features.keys()}'."
                )

            if self.features[key]["dtype"] in ["image", "video"]:
                img_path = self._get_image_file_path(
                    episode_index=self.episode_buffer["episode_index"], image_key=key, frame_index=frame_index
                )
                if frame_index == 0:
                    img_path.parent.mkdir(parents=True, exist_ok=True)
                self._save_image(frame[key], img_path)
                self.episode_buffer[key].append(str(img_path))
            else:
                self.episode_buffer[key].append(frame[key])

        self.episode_buffer["size"] += 1

    def save_episode(self, episode_data: dict | None = None) -> None:
        """
        This will save to disk the current episode in self.episode_buffer.

        Args:
            episode_data (dict | None, optional): Dict containing the episode data to save. If None, this will
                save the current episode in self.episode_buffer, which is filled with 'add_frame'. Defaults to
                None.
        """
        if not episode_data:
            episode_buffer = self.episode_buffer

        validate_episode_buffer(episode_buffer, self.meta.total_episodes, self.features)

        # size and task are special cases that won't be added to hf_dataset
        episode_length = episode_buffer.pop("size")
        tasks = episode_buffer.pop("task")
        episode_tasks = list(set(tasks))
        episode_index = episode_buffer["episode_index"]

        episode_buffer["index"] = np.arange(self.meta.total_frames, self.meta.total_frames + episode_length)
        episode_buffer["episode_index"] = np.full((episode_length,), episode_index)

        # Add new tasks to the tasks dictionary
        for task in episode_tasks:
            task_index = self.meta.get_task_index(task)
            if task_index is None:
                self.meta.add_task(task)

        # Given tasks in natural language, find their corresponding task indices
        episode_buffer["task_index"] = np.array([self.meta.get_task_index(task) for task in tasks])

        for key, ft in self.features.items():
            # index, episode_index, task_index are already processed above, and image and video
            # are processed separately by storing image path and frame info as meta data
            if key in ["index", "episode_index", "task_index"] or ft["dtype"] in ["image", "video"]:
                continue
            episode_buffer[key] = np.stack(episode_buffer[key])

        self._wait_image_writer()
        self._save_episode_table(episode_buffer, episode_index)
        ep_stats = compute_episode_stats(episode_buffer, self.features)

        if len(self.meta.video_keys) > 0:
            video_paths = self.encode_episode_videos(episode_index)
            for key in self.meta.video_keys:
                episode_buffer[key] = video_paths[key]

        # `meta.save_episode` be executed after encoding the videos
        self.meta.save_episode(episode_index, episode_length, episode_tasks, ep_stats)

        ep_data_index = get_episode_data_index(self.meta.episodes, [episode_index])
        ep_data_index_np = {k: t.numpy() for k, t in ep_data_index.items()}
        check_timestamps_sync(
            episode_buffer["timestamp"],
            episode_buffer["episode_index"],
            ep_data_index_np,
            self.fps,
            self.tolerance_s,
        )

        video_files = list(self.root.rglob("*.mp4"))
        assert len(video_files) == self.num_episodes * len(self.meta.video_keys)

        parquet_files = list(self.root.rglob("*.parquet"))
        assert len(parquet_files) == self.num_episodes

        # delete images
        img_dir = self.root / "images"
        if img_dir.is_dir():
            shutil.rmtree(self.root / "images")

        if not episode_data:  # Reset the buffer
            self.episode_buffer = self.create_episode_buffer()

    def _save_episode_table(self, episode_buffer: dict, episode_index: int) -> None:
        episode_dict = {key: episode_buffer[key] for key in self.hf_features}
        ep_dataset = datasets.Dataset.from_dict(episode_dict, features=self.hf_features, split="train")
        ep_dataset = embed_images(ep_dataset)
        self.hf_dataset = concatenate_datasets([self.hf_dataset, ep_dataset])
        self.hf_dataset.set_transform(hf_transform_to_torch)
        ep_data_path = self.root / self.meta.get_data_file_path(ep_index=episode_index)
        ep_data_path.parent.mkdir(parents=True, exist_ok=True)
        ep_dataset.to_parquet(ep_data_path)

    def clear_episode_buffer(self) -> None:
        episode_index = self.episode_buffer["episode_index"]
        if self.image_writer is not None:
            for cam_key in self.meta.camera_keys:
                img_dir = self._get_image_file_path(
                    episode_index=episode_index, image_key=cam_key, frame_index=0
                ).parent
                if img_dir.is_dir():
                    shutil.rmtree(img_dir)

        # Reset the buffer
        self.episode_buffer = self.create_episode_buffer()

    def start_image_writer(self, num_processes: int = 0, num_threads: int = 4) -> None:
        if isinstance(self.image_writer, AsyncImageWriter):
            logging.warning(
                "You are starting a new AsyncImageWriter that is replacing an already existing one in the dataset."
            )

        self.image_writer = AsyncImageWriter(
            num_processes=num_processes,
            num_threads=num_threads,
        )

    def stop_image_writer(self) -> None:
        """
        Whenever wrapping this dataset inside a parallelized DataLoader, this needs to be called first to
        remove the image_writer in order for the LeRobotDataset object to be pickleable and parallelized.
        """
        if self.image_writer is not None:
            self.image_writer.stop()
            self.image_writer = None

    def _wait_image_writer(self) -> None:
        """Wait for asynchronous image writer to finish."""
        if self.image_writer is not None:
            self.image_writer.wait_until_done()

    def encode_videos(self) -> None:
        """
        Use ffmpeg to convert frames stored as png into mp4 videos.
        Note: `encode_video_frames` is a blocking call. Making it asynchronous shouldn't speedup encoding,
        since video encoding with ffmpeg is already using multithreading.
        """
        for ep_idx in range(self.meta.total_episodes):
            self.encode_episode_videos(ep_idx)

    def encode_episode_videos(self, episode_index: int) -> dict:
        """
        Use ffmpeg to convert frames stored as png into mp4 videos.
        Note: `encode_video_frames` is a blocking call. Making it asynchronous shouldn't speedup encoding,
        since video encoding with ffmpeg is already using multithreading.
        """
        video_paths = {}
        for key in self.meta.video_keys:
            video_path = self.root / self.meta.get_video_file_path(episode_index, key)
            video_paths[key] = str(video_path)
            if video_path.is_file():
                # Skip if video is already encoded. Could be the case when resuming data recording.
                continue
            img_dir = self._get_image_file_path(
                episode_index=episode_index, image_key=key, frame_index=0
            ).parent
            encode_video_frames(img_dir, video_path, self.fps, overwrite=True)

        return video_paths

    @classmethod
    def create(
        cls,
        repo_id: str,
        fps: int,
        root: str | Path | None = None,
        robot: None = None,
        robot_type: str | None = None,
        features: dict | None = None,
        use_videos: bool = True,
        tolerance_s: float = 1e-4,
        image_writer_processes: int = 0,
        image_writer_threads: int = 0,
        video_backend: str | None = None,
    ) -> "LeRobotDataset":
        """Create a LeRobot Dataset from scratch in order to record data."""
        obj = cls.__new__(cls)
        obj.meta = LeRobotDatasetMetadata.create(
            repo_id=repo_id,
            fps=fps,
            root=root,
            robot=robot,
            robot_type=robot_type,
            features=features,
            use_videos=use_videos,
        )
        obj.repo_id = obj.meta.repo_id
        obj.root = obj.meta.root
        obj.revision = None
        obj.tolerance_s = tolerance_s
        obj.image_writer = None

        if image_writer_processes or image_writer_threads:
            obj.start_image_writer(image_writer_processes, image_writer_threads)

        # TODO(aliberts, rcadene, alexander-soare): Merge this with OnlineBuffer/DataBuffer
        obj.episode_buffer = obj.create_episode_buffer()

        obj.episodes = None
        obj.hf_dataset = obj.create_hf_dataset()
        obj.image_transforms = None
        obj.delta_timestamps = None
        obj.delta_indices = None
        obj.episode_data_index = None
        obj.video_backend = video_backend if video_backend is not None else "pyav"
        return obj


class MultiDatasetforDistTraining(torch.utils.data.Dataset):
    def __init__(
            self, 
            data_dir: str,
            chunk_size: int = 8,
            final_image_size: int = 224,
            t5_text_embeddings_path: str = "",
            normalize_images=False,
            normalize_actions=True,
            normalize_proprio=True,
            use_image_aug: bool = True,
            use_stronger_image_aug: bool = True,
            use_wrist_images: bool = True,
            use_third_person_images: bool = True,
            use_proprio: bool = True,
            num_duplicates_per_image: int = 4,
            rollout_data_dir: str = "",
            demonstration_sampling_prob: float = 0.5,
            success_rollout_sampling_prob: float = 0.5,
            treat_success_rollouts_as_demos: bool = False,
            return_value_function_returns: bool = True,
            gamma: float = 0.99,
            seed: int = 1001,
            stage: str = "finetune",
            data_mix: str = "libero",
            parent_dir: str = "",
            vla2root_json: str = "vla2root.json",
            balance_dataset_weights: bool = True,
            max_action_dim: int = 32,
            max_state_dim: int = 32,
            dataset_len_one_epoch = 5000_0000
        ):
        super().__init__()
        self.seed = seed
        self.stage = stage
        # 1. prepare mixture dataset
        data_mixture = OXE_NAMED_MIXTURES[data_mix]
        included_d_names = []
        dataset_sampling_weights = []
        for d_name, d_weight in data_mixture:
            if d_name in included_d_names:
                print(f"Skipping Duplicate Dataset: `{(d_name, d_weight)}`")
                continue

            included_d_names.append(d_name)
            dataset_sampling_weights.append(d_weight)
        
        # make dataset
        self.datasets = []
        self.dataset_sizes = []
        self.dataset_names = []
        self.num_episodes = 0
        self.num_frames = 0
        with open(vla2root_json, "r") as f:
            vla2data_root = json.load(f)
        for dataset_name in included_d_names:
            if dataset_name in vla2data_root.keys():
                data_root = vla2data_root[dataset_name]
                data_root = os.path.join(parent_dir, data_root)
                if os.path.exists(data_root):
                    print(f"Load data from {data_root}")
                    repo_id = f"bulldog-{dataset_name}" # any
                    ds_meta = LeRobotDatasetMetadata(repo_id, root=data_root)
                    delta_timestamps = resolve_delta_timestamps(ds_meta, chunk_size)
                    if self.stage == "pretrain":
                        image_transforms = v2.Resize((final_image_size, final_image_size))
                    else:
                        image_transforms = None
                    # image_transforms = v2.Resize((final_image_size, final_image_size))
                    dataset = LeRobotDataset(
                        repo_id, 
                        root=data_root,
                        delta_timestamps=delta_timestamps,
                        image_transforms=image_transforms,
                        wrist_image_transforms=None,
                        video_backend="torchcodec",
                        dataset_name=dataset_name,
                    )
                    self.num_episodes += dataset.num_episodes
                    self.num_frames += dataset.num_frames
                    self.datasets.append(dataset)
                    self.dataset_sizes.append(len(dataset))
                    self.dataset_names.append(dataset_name)
            else:
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] - {dataset_name} not found in vla2root.json, skipping...")

        # 2. Set properties for sampling
        # self.set_epoch(0)
        self.balance_dataset_weights = balance_dataset_weights
        self._dataset_lengths = np.array([len(dataset) for dataset in self.datasets])
        
        print(f"Dataset lengths: {self._dataset_lengths} Num episodes:{self.num_episodes}")

        # Dataset sampling weights
        self._dataset_sampling_weights = np.array(dataset_sampling_weights)
        
        if self.balance_dataset_weights:
            self._dataset_sampling_weights *= self._dataset_lengths
        
        # Normalize weights
        weights_sum = self._dataset_sampling_weights.sum()
        if weights_sum == 0 or np.isnan(weights_sum):
            print(f"Error: Invalid weights sum: {weights_sum}")
            # Fallback to equal weights
            self._dataset_sampling_weights = np.ones(len(self.datasets)) / len(self.datasets)
            print(f"Fallback to equal weights")
        else:
            self._dataset_sampling_weights /= weights_sum
        
        # table_data = [
        #     [self.dataset_names[i], len(self.datasets[i]), f"{self._dataset_sampling_weights[i]:.4f}"]
        #         for i in range(len(self.datasets))
        # ]
        # print(tabulate(table_data, headers=["Dataset", "Frames", "Ratio"], tablefmt="grid"))
        # print(f"Total frames: {self._dataset_lengths.sum()}")
        
        if self.stage == "pretrain":
            print(f"Building pretrain dataset with target size {dataset_len_one_epoch}...")
            self.target_size = dataset_len_one_epoch   # 例如固定 5w
            self.selected_indices, self.dataset_len = self.build_pretrain_dataset(
                target_size=self.target_size,
                seed=self.seed
            )
            self.dataset_ids = [i for i in range(len(self.datasets))]
            # self.dataset_len = 
        else:
            self.full_dataset = ConcatDataset(self.datasets)
            self.dataset_len = len(self.full_dataset)
            self.target_size = self.dataset_len
        
        print(f"Dataset Len:{self.dataset_len}")
        # 4. Aggregate dataset stats from all datasets
        self.stats = aggregate_stats([dataset.meta.stats for dataset in self.datasets], 
                                     max_dims = {"action": max_action_dim, "observation.state": max_state_dim})
        
        # save_to_json(self.stats, os.path.join("/home/cosmos/.cache/lerobot_data", f"{data_mix}_stats.json"))
        # save_to_json(self.stats, os.path.join("/mnt/wangxiaofa/robot_dataset/lerobot-format", f"{data_mix}_stats.json"))
        
        
        # in fact, we do not use it, so just simply copy
        self.meta = ds_meta
        
        if self.stage == "finetune":
            #
            # t5_text_embeddings_path = os.path.join("/mnt/wangxiaofa/robot_dataset/lerobot-format-v21-ort6d", f"t5_embeddings_{data_mix}.pkl")
            # for real world
            t5_text_embeddings_path = "/mnt/wangxiaofa/robot_dataset/lerobot-format-v21-ort6d/t5_embeddings/t5_embeddings_real_world.pkl"
        else:
            t5_text_embeddings_path = os.path.join(parent_dir, f"t5_embeddings_pretrain.pkl")
        if os.path.exists(t5_text_embeddings_path):
            with open(t5_text_embeddings_path, "rb") as file:
                self.t5_text_embeddings = pickle.load(file)
        self.t5_text_embeddings_dir = os.path.join(parent_dir, "t5_embeddings")
        
        # other property
        self.use_proprio = use_proprio
        self.use_wrist_images = use_wrist_images
        self.use_third_person_images = use_third_person_images
        self.num_duplicates_per_image = num_duplicates_per_image
        self.final_image_size = final_image_size
        self.normalize_images = normalize_images
        self.use_image_aug = use_image_aug
        self.use_stronger_image_aug = use_stronger_image_aug
        self.max_action_dim = max_action_dim
        self.max_state_dim = max_state_dim
    
    
    def build_pretrain_dataset(self, target_size, seed=0):
        rng = np.random.default_rng(seed)

        sample_counts = allocate_samples_by_weights(
            self._dataset_sampling_weights,
            target_size
        )

        # sampled_subsets = []
        sampled_table = []
        selected_indices = []

        for i, (dataset, count) in enumerate(zip(self.datasets, sample_counts)):
            ds_name = self.dataset_names[i]
            ds_len = len(dataset)

            if ds_len == 0:
                print(f"Warning: {ds_name} is empty, skip.")
                continue

            if count <= 0:
                print(f"Info: {ds_name} sampled count is 0, skip.")
                continue

            # 有放回采样
            indices = rng.integers(0, ds_len, size=count)
            selected_indices.append(indices.tolist())

            sampled_table.append([
                ds_name,
                ds_len,
                count,
                f"{count / target_size:.4f}"
            ])

        print(tabulate(
            sampled_table,
            headers=["Dataset", "Original Frames", "Sampled Frames", "Ratio"],
            tablefmt="grid"
        ))
        total_sample_len = sum(len(x) for x in selected_indices)
        print(f"Total sampled frames: {total_sample_len}")

        return selected_indices, total_sample_len
    
    def set_epoch(self, epoch):
        print(f"Setting epoch to {epoch}..., Update random seed for sampling.")
        self.epoch = epoch
        self.selected_indices, self.dataset_len = self.build_pretrain_dataset(
            target_size=self.target_size,
            seed=self.seed + self.epoch
        )
    
    def pad_vector(self, vector, new_dim):
        """Can be (batch_size x sequence_length x features_dimension)
        or (batch_size x features_dimension)
        """
        if vector.shape[-1] == new_dim:
            return vector
        shape = list(vector.shape)
        current_dim = shape[-1]
        shape[-1] = new_dim
        new_vector = torch.zeros(*shape, dtype=vector.dtype, device=vector.device)
        new_vector[..., :current_dim] = vector
        return new_vector
    
    def __len__(self):
        # return len(self.dataset)
        return self.dataset_len
    
    def sample_step(self, index):
        seed = safe_hash((self.epoch, index, self.seed))
        rng = np.random.default_rng(seed)

        dataset_index = rng.choice(
            len(self.datasets),
            p=self._dataset_sampling_weights
        )

        dataset = self.datasets[dataset_index]

        single_step_index = rng.integers(len(dataset))

        return dataset[int(single_step_index)]
    
    def prepare_action_state(self, item):
        if "game" in item["dataset_name"]:
            item["action"] = F.pad(
                    item["action"],
                    (44 + 6, 0),   # 对最后一维：左 pad 44个0，右 pad 0个0
                    mode="constant",
                    value=0
                )
            item["observation.state"] = F.pad(
                    item["observation.state"],
                    (46 + 6, 0),   # 对最后一维：左 pad 46个0，右 pad 0个0
                    mode="constant",
                    value=0
                )
        if "rh20t" in item["dataset_name"]:
            chunk_len = item["action"].shape[0]
            new_action = torch.ones((chunk_len, self.max_action_dim))
            new_action[:, :6] = item["action"][:, :6]
            new_action[:, 6:6 + 1] = item["action"][:, -2:-1]
            # force data
            new_action[:, 44:44 + 6] = item["action"][:, 6:6 + 6]
            new_state = torch.ones((chunk_len, self.max_state_dim))
            new_state[:, :7] = item["observation.state"][:, :7]
            new_state[:, 7:7 + 1] = item["observation.state"][:, -2:-1]
            # force data
            new_state[:, 46:46 + 6] = item["observation.state"][:, 7:7 + 6]
            item["action"] = new_action
            item["observation.state"] = new_state
        
        item["action"] = self.pad_vector(item["action"], self.max_action_dim)
        item["observation.state"] = self.pad_vector(item["observation.state"], self.max_state_dim)
        return item
    
    def norm_data_with_quantile(self, item):
        # key1 = "min"
        # key2 = "max"
        key1 = "min"
        key2 = "max"
        state_q01 = torch.ones(self.max_state_dim) * -1
        state_q99 = torch.ones(self.max_state_dim)
        action_q01 = torch.ones(self.max_action_dim) * -1
        action_q99 = torch.ones(self.max_action_dim)
        action_mask = torch.zeros(self.max_action_dim)
        action_start_dim = 0
        action_end_dim = 0
        state_start_dim = 0
        state_end_dim = 0
        if "agi" in item['dataset_name']:
            action_end_dim = 14
            state_end_dim = 16
        elif "ego_dex" in item['dataset_name']:
            action_start_dim = 0
            action_end_dim = 14 + 30
            state_start_dim = 0
            state_end_dim = 16 + 30
        elif "game" in item["dataset_name"]:
            action_start_dim = 14 + 30
            action_end_dim = 14 + 30 + 50
            state_start_dim = 16 + 30
            state_end_dim = 16 + 30 + 50
        else:
            action_end_dim = 7
            state_end_dim = 8
        
        state_q01[state_start_dim:state_end_dim] = self.stats["observation.state"][key1][state_start_dim:state_end_dim]
        state_q99[state_start_dim:state_end_dim] = self.stats["observation.state"][key2][state_start_dim:state_end_dim]
        action_q01[action_start_dim:action_end_dim] = self.stats["action"][key1][action_start_dim:action_end_dim]
        action_q99[action_start_dim:action_end_dim] = self.stats["action"][key2][action_start_dim:action_end_dim]
        # action
        denom = action_q99 - action_q01
        denom = torch.where(
            denom == 0, torch.tensor(1e-8), denom
        )
        item["action"] = 2.0 * (item["action"] - action_q01) / denom - 1.0
        
        # state
        denom = state_q99 - state_q01
        denom = torch.where(
            denom == 0, torch.tensor(1e-8), denom
        )
        item["observation.state"] = 2.0 * (item["observation.state"] - state_q01) / denom - 1.0
        
        # state_select_
        
        return item
    
    def __getitem__(self, index):
        # item = self.full_dataset[index]
        # every item key contains t-t+chunk_size elements (large than episode length use repeat last)
        if self.stage == "pretrain":
            dataset_id = random.choices(self.dataset_ids, weights=self._dataset_sampling_weights, k=1)[0]
            dataset = self.datasets[dataset_id]
            indices = self.selected_indices[dataset_id] # the selected indices of this dataset
            selected_id = random.choice(indices) # equal prob
            # selected_id = 0
            item = dataset[selected_id]
        else:
            item = self.full_dataset[index]
        
        # del item
        task_id = item["task_index"].item()
        dataset_name = item["dataset_name"]
        if self.stage == "pretrain":
            task_embeddings_path = os.path.join(self.t5_text_embeddings_dir, dataset_name, f"task_{task_id}.npy")
            with open(task_embeddings_path, 'rb') as f:
                # 不使用 mmap，读取后立即转为 Tensor 并 clone，断开与 numpy 的内存联系
                task_embeddings = torch.from_numpy(np.load(f)).squeeze()
                task_embeddings = torch.squeeze(task_embeddings)
            # task_embeddings = torch.squeeze(torch.from_numpy(np.load(task_embeddings_path)))
            # print(task_embeddings.shape)
            # task_embeddings = torch.squeeze(torch.zeros((1, 512, 1024)))
        else:
            # task_embeddings = torch.squeeze(self.t5_text_embeddings[item["task"]])
            task_embeddings = torch.from_numpy(self.t5_text_embeddings[item["task"]]).squeeze()
            # task_embeddings = torch.squeeze(task_embeddings)
        
        # prepare state and action
        item = self.prepare_action_state(item)
        item = self.norm_data_with_quantile(item) # follow cosmos policy
        
        # unified the image keys
        
        data_config = OXE_DATASET_CONFIGS[dataset_name]
        image_obs_keys = data_config["image_obs_keys"] # contain new_key: old_key mapping, such as "primary": "image", ...
        key_to_pad = []
        for new_key, old_key in image_obs_keys.items():
            if old_key != None:
                # print(item[f"observation.images.{old_key}"][0].shape)
                item[f"observation.images.{new_key}"] = item[f"observation.images.{old_key}"]
                exist_image = item[f"observation.images.{old_key}"]
                if new_key != old_key:
                    del item[f"observation.images.{old_key}"]
            else:
                # if missing, use zero image
                key_to_pad.append(new_key)
        
        for new_key in key_to_pad:
            item[f"observation.images.{new_key}"] = np.zeros_like(exist_image)
        
        # Prepare data for cosmos policy
        
        # Initialize list to store all images
        image_list = []
        current_sequence_idx = 0  # Used to track which sequence of images we are on
        # Get blank array for the first input frame (needed for the tokenizer)
        # Do not duplicate this image
        IMAGE_PRIMARY = "observation.images.primary"
        IMAGE_SECOND = "observation.images.secondary"
        IMAGE_WRIST = "observation.images.wrist"
        CURRENT_IDX = 0
        FUTURE_IDX = -1
        first_input_image = np.expand_dims(np.zeros_like(item[IMAGE_PRIMARY][CURRENT_IDX]), axis=0)
        image_list.append(first_input_image)
        current_sequence_idx += 1
        
        # current state
        if self.use_proprio:
            proprio = item[OBS_ROBOT][CURRENT_IDX]
            # Proprio values will be injected into latent diffusion sequence later
            # For now just add blank image
            blank_image = np.zeros_like(item[IMAGE_PRIMARY][CURRENT_IDX])
            blank_image = duplicate_array(blank_image, total_num_copies=self.num_duplicates_per_image)
            image_list.append(blank_image)
            current_proprio_latent_idx = current_sequence_idx
            current_sequence_idx += 1
        
        if self.use_wrist_images:
            wrist_image = item[IMAGE_WRIST][CURRENT_IDX]
            # Duplicate wrist image
            wrist_image = duplicate_array(wrist_image, total_num_copies=self.num_duplicates_per_image)
            image_list.append(wrist_image)
            current_wrist_image_latent_idx = current_sequence_idx
            current_sequence_idx += 1

        # Add current third-person image
        if self.use_third_person_images:
            current_primary_image = item[IMAGE_PRIMARY][CURRENT_IDX]
            current_primary_image = duplicate_array(current_primary_image, total_num_copies=self.num_duplicates_per_image)
            image_list.append(current_primary_image)
            current_image_latent_idx = current_sequence_idx
            current_sequence_idx += 1
            
            current_image2_latent_idx = -1
            current_secondary_image = item[IMAGE_SECOND][CURRENT_IDX]
            current_secondary_image = duplicate_array(current_secondary_image, total_num_copies=self.num_duplicates_per_image)
            image_list.append(current_secondary_image)
            current_image2_latent_idx = current_sequence_idx
            current_sequence_idx += 1
            
        # Add blank image for action chunk
        blank_image = np.zeros_like(item[IMAGE_PRIMARY][CURRENT_IDX])
        # Duplicate blank image
        blank_image = duplicate_array(blank_image, total_num_copies=self.num_duplicates_per_image)
        image_list.append(blank_image)
        action_latent_idx = current_sequence_idx
        current_sequence_idx += 1
        
        # future state
        
        # Add future proprio
        if self.use_proprio:
            future_proprio = item[OBS_ROBOT][FUTURE_IDX]
            # Not using proprio image; proprio values will be injected into latent diffusion sequence later
            # For now just add blank image
            blank_image = np.zeros_like(item[IMAGE_PRIMARY][FUTURE_IDX])
            blank_image = duplicate_array(blank_image, total_num_copies=self.num_duplicates_per_image)
            image_list.append(blank_image)
            future_proprio_latent_idx = current_sequence_idx
            current_sequence_idx += 1

        # Add future wrist image
        if self.use_wrist_images:
            future_wrist_image = item[IMAGE_WRIST][FUTURE_IDX]
            future_wrist_image = duplicate_array(future_wrist_image, total_num_copies=self.num_duplicates_per_image)
            image_list.append(future_wrist_image)
            future_wrist_image_latent_idx = current_sequence_idx
            current_sequence_idx += 1

        # Add future third-person image
        
        if self.use_third_person_images:
            future_primary_image = item[IMAGE_PRIMARY][FUTURE_IDX]
            future_primary_image = duplicate_array(future_primary_image, total_num_copies=self.num_duplicates_per_image)
            image_list.append(future_primary_image)
            future_image_latent_idx = current_sequence_idx
            current_sequence_idx += 1

            
            future_image2_latent_idx = -1
            future_secondary_image = item[IMAGE_SECOND][FUTURE_IDX]
            future_secondary_image = duplicate_array(future_secondary_image, total_num_copies=self.num_duplicates_per_image)
            image_list.append(future_secondary_image)
            future_image2_latent_idx = current_sequence_idx
            current_sequence_idx += 1
        
        # Stack images and preprocess
        images = np.concatenate(image_list, axis=0)
        # print(len(image_list), images.shape)
        images = preprocess_image(
            images,
            final_image_size=self.final_image_size,
            normalize_images=self.normalize_images,
            use_image_aug=self.use_image_aug,
            # use_image_aug=False,
            stronger_image_aug=self.use_stronger_image_aug,
        )
        # print(images.shape) # torch.Size([37, 3, 256, 256])
        action_chunk = item["action"] # pad with last action
        # print(torch.max(proprio), torch.min(proprio), proprio.shape)
        # print(images.shape, action_chunk.shape, proprio.shape, future_proprio.shape) # torch.Size([3, 37, 224, 224]) torch.Size([16, 32]) torch.Size([32]) torch.Size([32])
        # print(self.t5_text_embeddings.keys(), item["task"])
        # print(self.t5_text_embeddings[item["task"]].shape) # 1 512 1024
        sample_dict = {
            "dataset_name": dataset_name,
            "video": images,
            "actions": action_chunk,
            "t5_text_embeddings": task_embeddings,
            "t5_text_mask": torch.ones(512, dtype=torch.int64),  # Just copying what others have done in this codebase
            "fps": 16,  # Just set to some fixed value since we aren't generating videos anyway
            "padding_mask": torch.zeros(
                1, self.final_image_size, self.final_image_size
            ),  # Just copying what others have done in this codebase
            "image_size": self.final_image_size
            * torch.ones(
                4
            ),  # Just copying what others have done in this codebase; important because it shows up as model input
            "proprio": proprio if self.use_proprio else torch.zeros_like(item[OBS_ROBOT][CURRENT_IDX]),
            "future_proprio": future_proprio if self.use_proprio else torch.zeros_like(item[OBS_ROBOT][FUTURE_IDX]),
            "__key__": index,  # Unique sample identifier (required for callbacks)
            
            "rollout_data_mask": 0, # demonstration data
            "rollout_data_success_mask": 1,
            "world_model_sample_mask": 0,
            "value_function_sample_mask": 0,
            # "global_rollout_idx": global_rollout_idx,
            "action_latent_idx": action_latent_idx,
            "value_latent_idx": -1,
            "current_proprio_latent_idx": current_proprio_latent_idx if self.use_proprio else -1,
            "current_wrist_image_latent_idx": current_wrist_image_latent_idx if self.use_wrist_images else -1,
            "current_image_latent_idx": current_image_latent_idx if self.use_third_person_images else -1,
            "current_image2_latent_idx": current_image2_latent_idx if self.use_third_person_images else -1,
            "future_proprio_latent_idx": future_proprio_latent_idx if self.use_proprio else -1,
            "future_wrist_image_latent_idx": future_wrist_image_latent_idx if self.use_wrist_images else -1,
            "future_image_latent_idx": future_image_latent_idx if self.use_third_person_images else -1,
            "future_image2_latent_idx": future_image2_latent_idx if self.use_third_person_images else -1,
            "value_function_return": float("-100"),
            # "next_action_chunk": next_action_chunk,
            # "next_value_function_return": next_value_function_return,
        }
        
        return sample_dict
        # return 1
