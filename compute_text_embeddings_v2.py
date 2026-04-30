from transformers import T5EncoderModel, T5TokenizerFast
import torch
import json
import os
import math
import numpy as np
import pickle
from tqdm import tqdm
from cosmos_policy.datasets.lerobot.mixtures import OXE_NAMED_MIXTURES


def encode_t5_text_embeddings(t5_model, t5_tokenizer, prompts, max_length, device):
    if isinstance(prompts, str):
        prompts = [prompts]
    if not prompts:
        raise ValueError("The input prompt list is empty.")

    batch_encoding = t5_tokenizer.batch_encode_plus(
        prompts,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_length=True,
        return_offsets_mapping=False,
    )

    input_ids = batch_encoding.input_ids.to(device)
    attn_mask = batch_encoding.attention_mask.to(device)

    outputs = t5_model(input_ids=input_ids, attention_mask=attn_mask)

    encoded_text = outputs.last_hidden_state
    lengths = attn_mask.sum(dim=1).cpu()

    for batch_id in range(encoded_text.shape[0]):
        encoded_text[batch_id][lengths[batch_id]:] = 0

    return encoded_text


text_model_type = "t5"
torch_dtype = torch.float32
device = "cuda"
data_mix = "real_world"
max_length = 512
max_task_chunk = 40000

text_embeddings_dict = {}

save_root = "/mnt/wangxiaofa/robot_dataset/lerobot-format-v21-ort6d/t5_embeddings"
os.makedirs(save_root, exist_ok=True)

if text_model_type == "t5":
    ckpt_path = "/mnt/wangxiaofa/RDT_module_params/t5-11b"

    tokenizer = T5TokenizerFast.from_pretrained(
        ckpt_path,
        torch_dtype=torch_dtype
    )

    text_encoder = T5EncoderModel.from_pretrained(
        ckpt_path,
        torch_dtype=torch_dtype
    ).to(device)

    text_encoder.eval()

    val2root_json_path = "vla2root.json"
    data_root = "/mnt/wangxiaofa/robot_dataset/lerobot-format"

    mixture_sets = OXE_NAMED_MIXTURES[data_mix]
    process_datasets = []

    for d_name, d_weight in mixture_sets:
        process_datasets.append(d_name)

    with open(val2root_json_path, "r") as f:
        name2path_dict = json.load(f)

    for d_name in process_datasets:
        d_path = name2path_dict[d_name]
        data_path = os.path.join(data_root, d_path)

        if not os.path.exists(data_path):
            print(f"[Skip] path not exist: {data_path}")
            continue

        task_path = os.path.join(data_path, "meta", "tasks.jsonl")

        tasks = []
        with open(task_path, "r") as f:
            for line in f:
                d_dict = json.loads(line)
                tasks.append((d_dict["task_index"], d_dict["task"]))

        task_chunk_len = math.ceil(len(tasks) / max_task_chunk)

        for chunk_id in range(task_chunk_len):
            print(f"Processing {data_path}, Chunk: {chunk_id + 1}/{task_chunk_len}")

            start = chunk_id * max_task_chunk
            end = min((chunk_id + 1) * max_task_chunk, len(tasks))

            process_tasks = tasks[start:end]

            for t_id, task in tqdm(process_tasks):
                with torch.no_grad():
                    t5_embeddings = encode_t5_text_embeddings(
                        text_encoder,
                        tokenizer,
                        task,
                        max_length=max_length,
                        device=device
                    )

                t5_embeddings = t5_embeddings.cpu().numpy().astype(np.float16)

                # key 是 task，value 是对应的 t5 embedding
                text_embeddings_dict[task] = t5_embeddings

    save_path = os.path.join(save_root, f"t5_embeddings_{data_mix}.pkl")

    with open(save_path, "wb") as fp:
        pickle.dump(text_embeddings_dict, fp)

    print(f"Saved T5 embeddings dict to: {save_path}")