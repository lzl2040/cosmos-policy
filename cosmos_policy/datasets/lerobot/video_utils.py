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
import json
import logging
import subprocess
import warnings
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pyarrow as pa
import torch
import torchvision
from datasets.features.features import register_feature
from PIL import Image
import importlib
from threading import Lock
import fsspec
from torchcodec.decoders import VideoDecoder
import math
from typing import List
from concurrent.futures import ThreadPoolExecutor


class VideoDecoderCache:
    """Thread-safe cache for video decoders to avoid expensive re-initialization."""

    def __init__(self):
        self._cache: dict[str, tuple[Any, Any]] = {}
        self._lock = Lock()

    def get_decoder(self, video_path: str):
        """Get a cached decoder or create a new one."""
        if importlib.util.find_spec("torchcodec"):
            from torchcodec.decoders import VideoDecoder
        else:
            raise ImportError("torchcodec is required but not available.")

        video_path = str(video_path)

        with self._lock:
            if video_path not in self._cache:
                # file_handle = fsspec.open(video_path).__enter__()
                decoder = VideoDecoder(video_path, seek_mode="approximate")
                self._cache[video_path] = (decoder, video_path)

            return self._cache[video_path][0]

    def clear(self):
        """Clear the cache and close file handles."""
        with self._lock:
            for _, file_handle in self._cache.values():
                file_handle.close()
            self._cache.clear()

    def size(self) -> int:
        """Return the number of cached decoders."""
        with self._lock:
            return len(self._cache)


class FrameTimestampError(ValueError):
    """Helper error to indicate the retrieved timestamps exceed the queried ones"""

    pass


def get_safe_default_codec():
    if importlib.util.find_spec("torchcodec"):
        return "torchcodec"
    else:
        logging.warning(
            "'torchcodec' is not available in your platform, falling back to 'pyav' as a default decoder"
        )
        return "pyav"

def decode_video_frames(
    video_path: Path | str,
    timestamps: list[float],
    tolerance_s: float,
    backend: str | None = None,
    return_type: str = None,
    worker_count = 1
) -> torch.Tensor:
    """
    Decodes video frames using the specified backend.

    Args:
        video_path (Path): Path to the video file.
        timestamps (list[float]): List of timestamps to extract frames.
        tolerance_s (float): Allowed deviation in seconds for frame retrieval.
        backend (str, optional): Backend to use for decoding. Defaults to "torchcodec" when available in the platform; otherwise, defaults to "pyav"..

    Returns:
        torch.Tensor: Decoded frames.

    Currently supports torchcodec on cpu and pyav.
    """
    if backend is None:
        backend = get_safe_default_codec()
    if backend == "torchcodec":
        return decode_video_frames_torchcodec(video_path, timestamps, tolerance_s, return_type=return_type, worker_count=worker_count)
    elif backend in ["pyav", "video_reader"]:
        return decode_video_frames_torchvision(video_path, timestamps, tolerance_s, backend)
    else:
        raise ValueError(f"Unsupported video backend: {backend}")

def split_indices(indices: List[int], num_chunks: int) -> List[List[int]]:
    """Split a list of indices into approximately equal chunks."""
    chunk_size = len(indices) // num_chunks
    chunks = []

    for i in range(num_chunks - 1):
        chunks.append(indices[i * chunk_size:(i + 1) * chunk_size])

    # Last chunk may be slightly larger
    chunks.append(indices[(num_chunks - 1) * chunk_size:])
    return chunks

def convert_to_giventype(data: torch.Tensor, type: int = 1):
    npimg = data.cpu().numpy()
    # print(npimg.shape)
    if npimg.shape[0] in (1, 3, 4):
        npimg = np.transpose(npimg, (1, 2, 0))
    return Image.fromarray(npimg) if type else npimg

def decode_frame_torchcodec(
    raw_bytes: bytes,
    indices: List[int],
): 
    decoder = VideoDecoder(raw_bytes, seek_mode="approximate")
    frames = decoder.get_frames_at(indices)
    return frames


_default_decoder_cache = VideoDecoderCache()

def decode_video_frames_torchcodec(
    video_path: Path | str,
    timestamps: list[float],
    tolerance_s: float,
    log_loaded_timestamps: bool = False,
    decoder_cache: VideoDecoderCache | None = None,
    return_type: str = "numpy",
    worker_count = 1
) -> torch.Tensor:
    """Loads frames associated with the requested timestamps of a video using torchcodec.

    Args:
        video_path: Path to the video file.
        timestamps: List of timestamps to extract frames.
        tolerance_s: Allowed deviation in seconds for frame retrieval.
        log_loaded_timestamps: Whether to log loaded timestamps.
        decoder_cache: Optional decoder cache instance. Uses default if None.

    Note: Setting device="cuda" outside the main process, e.g. in data loader workers, will lead to CUDA initialization errors.

    Note: Video benefits from inter-frame compression. Instead of storing every frame individually,
    the encoder stores a reference frame (or a key frame) and subsequent frames as differences relative to
    that key frame. As a consequence, to access a requested frame, we need to load the preceding key frame,
    and all subsequent frames until reaching the requested frame. The number of key frames in a video
    can be adjusted during encoding to take into account decoding time and video size in bytes.
    """
    if decoder_cache is None:
        decoder_cache = _default_decoder_cache

    # with open(str(video_path), "rb") as f:
    #     raw_bytes = f.read()
    
    # Use cached decoder instead of creating new one each time
    decoder = decoder_cache.get_decoder(str(video_path))
    # decoder = VideoDecoder(raw_bytes, seek_mode="approximate")

    loaded_ts = []
    loaded_frames = []

    # get metadata for frame information
    metadata = decoder.metadata
    average_fps = metadata.average_fps
    # convert timestamps to frame indices
    # timestamps = [timestamps[0], timestamps[-1]]
    num_frames = decoder._num_frames

    frame_indices = [
        min(round(ts * average_fps), num_frames - 1)
        for ts in timestamps
    ]
    frame_indices = [frame_indices[0], frame_indices[-1]]
    # frame_indices = [round(ts * average_fps) for ts in timestamps]
    # frame_indices = [frame_indices[0], frame_indices[-1]]
    # retrieve frames based on indices
    frames_batch = decoder.get_frames_at(indices=frame_indices)

    for frame, pts in zip(frames_batch.data, frames_batch.pts_seconds, strict=True):
        loaded_frames.append(frame)
        loaded_ts.append(pts.item())
        if log_loaded_timestamps:
            logging.info(f"Frame loaded at timestamp={pts:.4f}")

    # query_ts = torch.tensor(timestamps)
    # loaded_ts = torch.tensor(loaded_ts)
    # print(len(loaded_ts), query_ts)

    # compute distances between each query timestamp and loaded timestamps
    # dist = torch.cdist(query_ts[:, None], loaded_ts[:, None], p=1)
    # min_, argmin_ = dist.min(1)

    # is_within_tol = min_ < tolerance_s
    # if not is_within_tol.all():
    #     raise FrameTimestampError(
    #         f"One or several query timestamps unexpectedly violate the tolerance ({min_[~is_within_tol]} > {tolerance_s=})."
    #         " It means that the closest frame that can be loaded from the video is too far away in time."
    #         " This might be due to synchronization issues with timestamps during data collection."
    #         " To be safe, we advise to ignore this item during training."
    #         f"\nqueried timestamps: {query_ts}"
    #         f"\nloaded timestamps: {loaded_ts}"
    #         f"\nvideo: {video_path}"
    #     )

    # get closest frames to the query timestamps
    # closest_frames = torch.stack([loaded_frames[idx] for idx in argmin_])
    # closest_ts = loaded_ts[argmin_]

    # if log_loaded_timestamps:
    #     logging.info(f"{closest_ts=}")

    # # convert to float32 in [0,1] range
    # # closest_frames = (closest_frames / 255.0).type(torch.float32) # cosmos policy not need it
    # closest_frames = closest_frames.type(torch.float32)

    # if not len(timestamps) == len(closest_frames): 
    #     raise FrameTimestampError(
    #         f"Retrieved timestamps differ from queried {set(closest_frames) - set(timestamps)}"
    #     )
    
    # if worker_count == 1:
    #     frames = decoder.get_frames_at(frame_indices)
    #     loaded_frames = frames.data
    # else:
    #     chunks = split_indices(frame_indices, num_chunks=worker_count)
    #     results = []
    #     with ThreadPoolExecutor(max_workers=worker_count) as executor:
    #         futures = [
    #             executor.submit(decode_frame_torchcodec, raw_bytes, chunk) 
    #             for chunk in chunks
    #         ]
    #         for f in futures:
    #             results.append(f.result())
    #         loaded_frames = torch.cat([frame_batch.data for frame_batch in results], dim=0)
    
    if return_type == "tensor":
        if log_loaded_timestamps:
            logging.info(f"{loaded_ts=}")
        
        return loaded_frames
    elif return_type == "image":
        image_list = []
        for idx in range(len(loaded_frames)):
            img = Image.fromarray(loaded_frames[idx].numpy().astype(np.uint8).transpose(1, 2, 0))
            image_list.append(img)
        return image_list
    elif return_type == "numpy":
        image_list = []
        for idx in range(len(loaded_frames)):
            img = loaded_frames[idx].numpy().astype(np.uint8).transpose(1, 2, 0)
            image_list.append(img)
        return image_list

def decode_video_frames_torchvision_org(
    video_path: Path | str,
    timestamps: list[float],
    tolerance_s: float,
    backend: str = "pyav",
    log_loaded_timestamps: bool = False,
) -> torch.Tensor:
    """Loads frames associated to the requested timestamps of a video

    The backend can be either "pyav" (default) or "video_reader".
    "video_reader" requires installing torchvision from source, see:
    https://github.com/pytorch/vision/blob/main/torchvision/csrc/io/decoder/gpu/README.rst
    (note that you need to compile against ffmpeg<4.3)

    While both use cpu, "video_reader" is supposedly faster than "pyav" but requires additional setup.
    For more info on video decoding, see `benchmark/video/README.md`

    See torchvision doc for more info on these two backends:
    https://pytorch.org/vision/0.18/index.html?highlight=backend#torchvision.set_video_backend

    Note: Video benefits from inter-frame compression. Instead of storing every frame individually,
    the encoder stores a reference frame (or a key frame) and subsequent frames as differences relative to
    that key frame. As a consequence, to access a requested frame, we need to load the preceding key frame,
    and all subsequent frames until reaching the requested frame. The number of key frames in a video
    can be adjusted during encoding to take into account decoding time and video size in bytes.
    """
    video_path = str(video_path)

    # set backend
    keyframes_only = False
    torchvision.set_video_backend(backend)
    if backend == "pyav":
        keyframes_only = True  # pyav doesn't support accurate seek

    # set a video stream reader
    # TODO(rcadene): also load audio stream at the same time
    reader = torchvision.io.VideoReader(video_path, "video")

    # set the first and last requested timestamps
    # Note: previous timestamps are usually loaded, since we need to access the previous key frame
    first_ts = min(timestamps)
    last_ts = max(timestamps)

    # access closest key frame of the first requested frame
    # Note: closest key frame timestamp is usually smaller than `first_ts` (e.g. key frame can be the first frame of the video)
    # for details on what `seek` is doing see: https://pyav.basswood-io.com/docs/stable/api/container.html?highlight=inputcontainer#av.container.InputContainer.seek
    reader.seek(first_ts, keyframes_only=keyframes_only)

    # load all frames until last requested frame
    loaded_frames = []
    loaded_ts = []
    for frame in reader:
        current_ts = frame["pts"]
        if log_loaded_timestamps:
            logging.info(f"frame loaded at timestamp={current_ts:.4f}")
        loaded_frames.append(frame["data"])
        loaded_ts.append(current_ts)
        if current_ts >= last_ts:
            break

    if backend == "pyav":
        reader.container.close()

    reader = None

    query_ts = torch.tensor(timestamps)
    loaded_ts = torch.tensor(loaded_ts)

    # compute distances between each query timestamp and timestamps of all loaded frames
    dist = torch.cdist(query_ts[:, None], loaded_ts[:, None], p=1)
    min_, argmin_ = dist.min(1)

    is_within_tol = min_ < tolerance_s
    assert is_within_tol.all(), (
        f"One or several query timestamps unexpectedly violate the tolerance ({min_[~is_within_tol]} > {tolerance_s=})."
        "It means that the closest frame that can be loaded from the video is too far away in time."
        "This might be due to synchronization issues with timestamps during data collection."
        "To be safe, we advise to ignore this item during training."
        f"\nqueried timestamps: {query_ts}"
        f"\nloaded timestamps: {loaded_ts}"
        f"\nvideo: {video_path}"
        f"\nbackend: {backend}"
    )

    # get closest frames to the query timestamps
    closest_frames = torch.stack([loaded_frames[idx] for idx in argmin_])
    closest_ts = loaded_ts[argmin_]

    if log_loaded_timestamps:
        logging.info(f"{closest_ts=}")

    # convert to the pytorch format which is float32 in [0,1] range (and channel first)
    closest_frames = closest_frames.type(torch.float32) / 255

    assert len(timestamps) == len(closest_frames)
    return closest_frames

def decode_video_frames_torchvision(
    video_path: Path | str,
    timestamps: list[float],
    tolerance_s: float,
    backend: str = "pyav",
    log_loaded_timestamps: bool = False,
    return_all: bool = False,
    return_type: str = "tensor",
    max_frame_window: int = -1
) -> torch.Tensor:
    """Loads frames associated to the requested timestamps of a video

    The backend can be either "pyav" (default) or "video_reader".
    "video_reader" requires installing torchvision from source, see:
    https://github.com/pytorch/vision/blob/main/torchvision/csrc/io/decoder/gpu/README.rst
    (note that you need to compile against ffmpeg<4.3)

    While both use cpu, "video_reader" is supposedly faster than "pyav" but requires additional setup.
    For more info on video decoding, see `benchmark/video/README.md`

    See torchvision doc for more info on these two backends:
    https://pytorch.org/vision/0.18/index.html?highlight=backend#torchvision.set_video_backend

    Note: Video benefits from inter-frame compression. Instead of storing every frame individually,
    the encoder stores a reference frame (or a key frame) and subsequent frames as differences relative to
    that key frame. As a consequence, to access a requested frame, we need to load the preceding key frame,
    and all subsequent frames until reaching the requested frame. The number of key frames in a video
    can be adjusted during encoding to take into account decoding time and video size in bytes.
    """
    video_path = str(video_path)

    # set backend
    keyframes_only = False
    torchvision.set_video_backend(backend)
    if backend == "pyav":
        keyframes_only = True  # pyav doesnt support accuracte seek

    # set a video stream reader
    # TODO(rcadene): also load audio stream at the same time
    reader = torchvision.io.VideoReader(video_path, "video")

    # set the first and last requested timestamps
    # Note: previous timestamps are usually loaded, since we need to access the previous key frame
    first_ts = min(timestamps)
    # 这个会从最开始一帧算
    # if return_all:
    #     first_ts = min(first_ts, 0.0)
    last_ts = max(timestamps)

    # access closest key frame of the first requested frame
    # Note: closest key frame timestamp is usually smaller than `first_ts` (e.g. key frame can be the first frame of the video)
    # for details on what `seek` is doing see: https://pyav.basswood-io.com/docs/stable/api/container.html?highlight=inputcontainer#av.container.InputContainer.seek
    reader.seek(first_ts, keyframes_only=keyframes_only)

    # load all frames until last requested frame
    loaded_frames = []
    loaded_ts = []

    # 只返回当前帧
    if max_frame_window == -1:
        # 读到 last_ts
        for frame in reader:
            current_ts = frame["pts"]
            if log_loaded_timestamps:
                logging.info(f"frame loaded at timestamp={current_ts:.4f}")
            loaded_frames.append(frame["data"])
            loaded_ts.append(current_ts)
            if current_ts >= last_ts:
                break
    else:
        # 从first_ts往后取max_frame_window帧
        try:
            for frame in reader:
                current_ts = frame["pts"]
                # if current_ts < first_ts:
                #     continue
                if len(loaded_frames) >= max_frame_window:
                    break
                if log_loaded_timestamps:
                    logging.info(f"frame loaded at timestamp={current_ts:.4f}")
                loaded_frames.append(frame["data"])
                loaded_ts.append(current_ts)
                # print(current_ts, first_ts, len(loaded_frames))
        except Exception as e:
            print(f"Frame decode error: {e} from {video_path} using fallback ones tensor.")
        
        # reader_iter = iter(reader)

        # for _ in range(max_frame_window):
        #     try:
        #         frame = next(reader_iter)   # 关键：显式 next 才能捕获异常
        #         current_ts = frame['pts']
        #         tensor = frame["data"]  # (C,H,W)

        #         if log_loaded_timestamps:
        #             logging.info(f"frame loaded at timestamp={current_ts:.4f}")

        #         loaded_frames.append(frame["data"])
        #         loaded_ts.append(current_ts)

        #     except StopIteration:
        #         # 视频已读完，直接退出，不补帧
        #         logging.info("Video finished early. Stop reading.")
        #         break

        #     except Exception as e:
        #         # 解码错误时：补一个全 1 tensor
        #         logging.warning(f"Frame decode error: {e} from {video_path} using fallback ones tensor.")
        #         print(f"Frame decode error: {e} from {video_path} using fallback ones tensor.")
        #         if len(loaded_frames) > 0:
        #             ones_frame = torch.ones_like(loaded_frames[0])
        #         else:
        #             # 如果第一帧就坏，必须给定 frame_shape
        #             meta = reader.get_metadata()
        #             w = meta["video"]["width"]
        #             h = meta["video"]["height"]
        #             ones_frame = torch.ones((3, h, w), dtype=torch.uint8)

        #         loaded_frames.append(ones_frame)
        #         loaded_ts.append(-1)

    if backend == "pyav":
        reader.container.close()

    reader = None

    if not return_all:
        query_ts = torch.tensor(timestamps)
        loaded_ts = torch.tensor(loaded_ts)

        # compute distances between each query timestamp and timestamps of all loaded frames
        dist = torch.cdist(query_ts[:, None], loaded_ts[:, None], p=1)
        min_, argmin_ = dist.min(1)

        is_within_tol = min_ < tolerance_s
        assert is_within_tol.all(), (
            f"One or several query timestamps unexpectedly violate the tolerance ({min_[~is_within_tol]} > {tolerance_s=})."
            "It means that the closest frame that can be loaded from the video is too far away in time."
            "This might be due to synchronization issues with timestamps during data collection."
            "To be safe, we advise to ignore this item during training."
            f"\nqueried timestamps: {query_ts}"
            f"\nloaded timestamps: {loaded_ts}"
            f"\nvideo: {video_path}"
            f"\nbackend: {backend}"
        )

        if return_type == "tensor":
            # get closest frames to the query timestamps
            closest_frames = torch.stack([loaded_frames[idx] for idx in argmin_])
            closest_ts = loaded_ts[argmin_]

            if log_loaded_timestamps:
                logging.info(f"{closest_ts=}")

            # convert to the pytorch format which is float32 in [0,1] range (and channel first)
            # closest_frames = closest_frames.type(torch.float32) / 255
            assert len(timestamps) == len(closest_frames)
            
            return closest_frames
        elif return_type == "image":
            image_list = []
            for idx in argmin_:
                img = Image.fromarray(loaded_frames[idx].numpy().astype(np.uint8).transpose(1, 2, 0))
                image_list.append(img)
            return image_list
        elif return_type == "numpy":
            image_list = []
            for idx in argmin_:
                img = loaded_frames[idx].numpy().astype(np.uint8).transpose(1, 2, 0)
                image_list.append(img)
            return image_list
    else:
        if return_type == "tensor":
            loaded_ts = torch.tensor(loaded_ts)
            closest_frames = torch.stack(loaded_frames)
            # closest_frames = closest_frames.type(torch.float32) / 255
            
            if log_loaded_timestamps:
                logging.info(f"{loaded_ts=}")
            
            return closest_frames
        elif return_type == "image":
            image_list = []
            for idx in range(len(loaded_frames)):
                img = Image.fromarray(loaded_frames[idx].numpy().astype(np.uint8).transpose(1, 2, 0))
                image_list.append(img)
            return image_list
        elif return_type == "numpy":
            image_list = []
            for idx in range(len(loaded_frames)):
                img = loaded_frames[idx].numpy().astype(np.uint8).transpose(1, 2, 0)
                image_list.append(img)
            return image_list

def encode_video_frames(
    imgs_dir: Path | str,
    video_path: Path | str,
    fps: int,
    vcodec: str = "libsvtav1",
    pix_fmt: str = "yuv420p",
    g: int | None = 2,
    crf: int | None = 30,
    fast_decode: int = 0,
    log_level: str | None = "error",
    overwrite: bool = False,
) -> None:
    """More info on ffmpeg arguments tuning on `benchmark/video/README.md`"""
    video_path = Path(video_path)
    video_path.parent.mkdir(parents=True, exist_ok=True)

    ffmpeg_args = OrderedDict(
        [
            ("-f", "image2"),
            ("-r", str(fps)),
            ("-i", str(imgs_dir / "frame_%06d.png")),
            ("-vcodec", vcodec),
            ("-pix_fmt", pix_fmt),
        ]
    )

    if g is not None:
        ffmpeg_args["-g"] = str(g)

    if crf is not None:
        ffmpeg_args["-crf"] = str(crf)

    if fast_decode:
        key = "-svtav1-params" if vcodec == "libsvtav1" else "-tune"
        value = f"fast-decode={fast_decode}" if vcodec == "libsvtav1" else "fastdecode"
        ffmpeg_args[key] = value

    if log_level is not None:
        ffmpeg_args["-loglevel"] = str(log_level)

    ffmpeg_args = [item for pair in ffmpeg_args.items() for item in pair]
    if overwrite:
        ffmpeg_args.append("-y")

    ffmpeg_cmd = ["ffmpeg"] + ffmpeg_args + [str(video_path)]
    # redirect stdin to subprocess.DEVNULL to prevent reading random keyboard inputs from terminal
    subprocess.run(ffmpeg_cmd, check=True, stdin=subprocess.DEVNULL)

    if not video_path.exists():
        raise OSError(
            f"Video encoding did not work. File not found: {video_path}. "
            f"Try running the command manually to debug: `{''.join(ffmpeg_cmd)}`"
        )


@dataclass
class VideoFrame:
    # TODO(rcadene, lhoestq): move to Hugging Face `datasets` repo
    """
    Provides a type for a dataset containing video frames.

    Example:

    ```python
    data_dict = [{"image": {"path": "videos/episode_0.mp4", "timestamp": 0.3}}]
    features = {"image": VideoFrame()}
    Dataset.from_dict(data_dict, features=Features(features))
    ```
    """

    pa_type: ClassVar[Any] = pa.struct({"path": pa.string(), "timestamp": pa.float32()})
    _type: str = field(default="VideoFrame", init=False, repr=False)

    def __call__(self):
        return self.pa_type


with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        "'register_feature' is experimental and might be subject to breaking changes in the future.",
        category=UserWarning,
    )
    # to make VideoFrame available in HuggingFace `datasets`
    register_feature(VideoFrame, "VideoFrame")


def get_audio_info(video_path: Path | str) -> dict:
    ffprobe_audio_cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=channels,codec_name,bit_rate,sample_rate,bit_depth,channel_layout,duration",
        "-of",
        "json",
        str(video_path),
    ]
    result = subprocess.run(ffprobe_audio_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Error running ffprobe: {result.stderr}")

    info = json.loads(result.stdout)
    audio_stream_info = info["streams"][0] if info.get("streams") else None
    if audio_stream_info is None:
        return {"has_audio": False}

    # Return the information, defaulting to None if no audio stream is present
    return {
        "has_audio": True,
        "audio.channels": audio_stream_info.get("channels", None),
        "audio.codec": audio_stream_info.get("codec_name", None),
        "audio.bit_rate": int(audio_stream_info["bit_rate"]) if audio_stream_info.get("bit_rate") else None,
        "audio.sample_rate": int(audio_stream_info["sample_rate"])
        if audio_stream_info.get("sample_rate")
        else None,
        "audio.bit_depth": audio_stream_info.get("bit_depth", None),
        "audio.channel_layout": audio_stream_info.get("channel_layout", None),
    }


def get_video_info(video_path: Path | str) -> dict:
    ffprobe_video_cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=r_frame_rate,width,height,codec_name,nb_frames,duration,pix_fmt",
        "-of",
        "json",
        str(video_path),
    ]
    result = subprocess.run(ffprobe_video_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Error running ffprobe: {result.stderr}")

    info = json.loads(result.stdout)
    video_stream_info = info["streams"][0]

    # Calculate fps from r_frame_rate
    r_frame_rate = video_stream_info["r_frame_rate"]
    num, denom = map(int, r_frame_rate.split("/"))
    fps = num / denom

    pixel_channels = get_video_pixel_channels(video_stream_info["pix_fmt"])

    video_info = {
        "video.fps": fps,
        "video.height": video_stream_info["height"],
        "video.width": video_stream_info["width"],
        "video.channels": pixel_channels,
        "video.codec": video_stream_info["codec_name"],
        "video.pix_fmt": video_stream_info["pix_fmt"],
        "video.is_depth_map": False,
        **get_audio_info(video_path),
    }

    return video_info


def get_video_pixel_channels(pix_fmt: str) -> int:
    if "gray" in pix_fmt or "depth" in pix_fmt or "monochrome" in pix_fmt:
        return 1
    elif "rgba" in pix_fmt or "yuva" in pix_fmt:
        return 4
    elif "rgb" in pix_fmt or "yuv" in pix_fmt:
        return 3
    else:
        raise ValueError("Unknown format")


def get_image_pixel_channels(image: Image):
    if image.mode == "L":
        return 1  # Grayscale
    elif image.mode == "LA":
        return 2  # Grayscale + Alpha
    elif image.mode == "RGB":
        return 3  # RGB
    elif image.mode == "RGBA":
        return 4  # RGBA
    else:
        raise ValueError("Unknown format")
