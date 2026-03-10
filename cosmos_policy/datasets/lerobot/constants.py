# keys
import os
from pathlib import Path

from huggingface_hub.constants import HF_HOME

OBS_ENV = "observation.environment_state"
OBS_ROBOT = "observation.state"
OBS_IMAGE = "observation.image"
OBS_IMAGES = "observation.images"
ACTION = "action"
OBS_LANGUAGE = "observation.language"
OBS_LANGUAGE_TOKENS = OBS_LANGUAGE + ".tokens"
OBS_LANGUAGE_ATTENTION_MASK = OBS_LANGUAGE + ".attention_mask"

# files & directories
CHECKPOINTS_DIR = "checkpoints"
LAST_CHECKPOINT_LINK = "last"
PRETRAINED_MODEL_DIR = "pretrained_model"
TRAINING_STATE_DIR = "training_state"
RNG_STATE = "rng_state.safetensors"
TRAINING_STEP = "training_step.json"
OPTIMIZER_STATE = "optimizer_state.safetensors"
OPTIMIZER_PARAM_GROUPS = "optimizer_param_groups.json"
SCHEDULER_STATE = "scheduler_state.json"

# cache dir
default_cache_path = Path(HF_HOME) / "lerobot"
HF_LEROBOT_HOME = Path(os.getenv("HF_LEROBOT_HOME", default_cache_path)).expanduser()

if "LEROBOT_HOME" in os.environ:
    raise ValueError(
        f"You have a 'LEROBOT_HOME' environment variable set to '{os.getenv('LEROBOT_HOME')}'.\n"
        "'LEROBOT_HOME' is deprecated, please use 'HF_LEROBOT_HOME' instead."
    )


# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
IMG_START_TOKEN = '<img>'
IMG_END_TOKEN = '</img>'
QUAD_START_TOKEN = '<quad>'
QUAD_END_TOKEN = '</quad>'
REF_START_TOKEN = '<ref>'
REF_END_TOKEN = '</ref>'
BOX_START_TOKEN = '<box>'
BOX_END_TOKEN = '</box>'
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLIP_MEAN = (0.4814546, 0.4578275, 0.40821073)
CLIP_STD = (0.2686295, 0.2613025, 0.2757711)
SIGLIP_MEAN = (0.5, 0.5, 0.5)
SIGLIP_STD = (0.5, 0.5, 0.5)

COMPRESS_SC_TOKEN = 'CP_SC'
COMPRESS_ACTION_TOKEN = 'CP_ACT'

# ANSWER_LIST = [
#     "Furture scene representations: [CP_SC]. Action representations: [CP_ACT].",
#     "The results for the next-frame scene are [CP_SC], while the motion is represented as [CP_ACT].",
# ]

# QUESTION_LIST = [
#     "According to the instruction '{sent}', compress the video into scene-level and action-level representations.",
#     "Observe the video frames under the task '{sent}', and summarize them into scene-level representations and action-level representations.",
# ]

# QUESTION_LIST = [
#     "Given the instruction '{sent}', predict the next-frame scene-level representation from the historical video, and compress the historical video into an action-level representation.",
#     "Based on the task '{sent}', use the historical video to forecast the next-frame scene-level representation, and condense the historical video into an action-level representation.",
# ]

# QUESTION_LIST = [
#     "Based on the instruction '{sent}', analyze the given {T} consecutive video frames.\n1. Forecast the next-frame scene-level representation.\n2. Generate {Tm1} action-level representations, one for each transition between consecutive frames.",
#     "Given the task '{sent}', process the {T} frames of video input as follows:\n1. Predict the scene representation for the upcoming frame.\n2. Produce {Tm1} action embeddings summarizing the transitions between adjacent frames.",
#     "Task: '{sent}'. You are provided with {T} continuous frames.\n1. Output the scene-level embedding that corresponds to the next frame.\n2. Output {Tm1} action-level embeddings, each describing one frame-to-frame transition.",
#     "Instruction: '{sent}'. Analyze the sequence of {T} frames.\n1. Forecast the representation of the scene in the next frame.\n2. Generate {Tm1} action embeddings to represent the motion between consecutive frames.",
# ]

# ip_adapter
QUESTION_LIST = [
    # 1. 通用生成型
    "You are given an instruction '{sent}' and a sequence of video frames. Analyze the visual content across these frames to understand how the scene evolves over time. Describe both the scene-level evolution—how the global environment and context change—and the action-level dynamics, detailing how objects or agents move and interact between consecutive frames.",
    # 2. Reasoning 型
    "Given the instruction '{sent}' and the video frames, reason about what happens within this time span. Explain how the overall scene changes, and identify the temporal dependencies between consecutive frames. Highlight the actions, interactions, and transitions that drive the scene’s evolution.",
    # 3. Summarization 型
    "According to the instruction '{sent}', generate a structured summary of the video segment. Describe the scene-level context and its temporal evolution, followed by the action-level motion between frames. Your summary should clearly separate static scene changes from dynamic object behaviors, providing a coherent overview of the visual events in this segment.",
    # 4. Embedding / Representation 型
    "Based on the instruction '{sent}', encode the video segment into hierarchical representations. The scene-level representation should summarize the overall visual environment and its evolution. The action-level representation should capture motion transitions and object interactions between adjacent frames. Together, they should form a compact embedding that reflects both global context and local temporal dynamics.",
]

ANSWER_LIST = [
    # 1️⃣ 通用生成型 —— 直接分析场景与动作变化
    "Scene evolution description: [CP_SC]. Action dynamics description: [CP_ACT].",
    # 2️⃣ Reasoning 型 —— 体现逻辑、因果与时间推理
    "Scene reasoning: [CP_SC]. Action reasoning and temporal dependencies: [CP_ACT].",
    # 3️⃣ Summarization 型 —— 生成结构化摘要
    "Scene summary: [CP_SC]. Action summary of motion and interactions: [CP_ACT].",
    # 4️⃣ Embedding / Representation 型 —— 输出层次化表征
    "Scene-level embedding: [CP_SC]. Action-level embedding capturing motion transitions: [CP_ACT]."
]

# openpi
OPENPI_ATTENTION_MASK_VALUE = -2.3819763e38  # TODO(pepijn): Modify this when extending support to fp8 models
