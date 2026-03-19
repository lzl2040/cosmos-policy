from transformers import T5EncoderModel, T5TokenizerFast
import torch
import json
import os
import math
import numpy as np
import pickle
from tqdm import tqdm
from cosmos_policy.datasets.lerobot.mixtures import OXE_NAMED_MIXTURES
# cosmos-predict-2
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
    # print(input_ids.shape) # 10 512
    outputs = t5_model(input_ids=input_ids, attention_mask=attn_mask)

    encoded_text = outputs.last_hidden_state
    lengths = attn_mask.sum(dim=1).cpu()

    for batch_id in range(encoded_text.shape[0]):
        encoded_text[batch_id][lengths[batch_id] :] = 0
    return encoded_text

text_model_type = "t5"
torch_dtype = torch.float32
device = "cuda"
data_mix = "oxe_magic_soup_plus"
process_chunk_size = 1
max_length = 512
hidden_size = 1024
max_task_chunk = 40000
text_embeddings_dict = {}
save_root = "/mnt/wangxiaofa/robot_dataset/lerobot-format/t5_embeddings"
os.makedirs(save_root, exist_ok=True)
if text_model_type == "t5":
    # ckpt_path = "/Data/lzl/huggingface/t5-11b"
    ckpt_path = "/mnt/wangxiaofa/RDT_module_params/t5-11b"
    tokenizer = T5TokenizerFast.from_pretrained(ckpt_path, torch_dtype=torch_dtype)
    text_encoder = T5EncoderModel.from_pretrained(ckpt_path, torch_dtype=torch_dtype).to(device)
    text_encoder.eval()
    val2root_json_path = "vla2root.json"
    # data_root = "/Data/lerobot_data"
    data_root = "/mnt/wangxiaofa/robot_dataset/lerobot-format"
    mixture_sets = OXE_NAMED_MIXTURES[data_mix]
    process_datasets = []
    for d_name, d_weight in mixture_sets:
        process_datasets.append(d_name, )
        
    with open(val2root_json_path, "r") as f:
        name2path_dict = json.load(f)
    
    # start_from = "language_table"
    # start_run = start_from is None

    for d_name in process_datasets:
        # if d_name == start_from:
        #     start_run = True
        # if not start_run:
        #     continue
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
        chunk_id = 0
        while chunk_id < task_chunk_len:
            print(f"Processing {data_path}, Chunk:{chunk_id}/{task_chunk_len}")
            start = chunk_id * max_task_chunk
            end = (chunk_id + 1) * max_task_chunk if (chunk_id + 1) * max_task_chunk < len(tasks) else len(tasks)
            process_tasks = tasks[start:end]
            text_embeddings = []
            for i in tqdm(range(len(process_tasks))):
                t_id, prompts = process_tasks[i]
                with torch.no_grad():
                    encoded_text = encode_t5_text_embeddings(text_encoder, tokenizer, prompts, 
                                                            max_length=max_length, device=device)
                encoded_text = encoded_text.cpu().numpy().astype(np.float16)
                save_dir = os.path.join(save_root, d_name)
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f"task_{t_id}.npy")
                np.save(save_path, encoded_text)
            chunk_id += 1
                
                # text_embeddings[start:end] = encoded_text
            #     text_embeddings.append(encoded_text)
            # save_path = os.path.join(save_root, f"t5_embeddings_{d_name}_chunk_{chunk_id}.pkl")
            # with open(save_path, "wb") as fp:
            #     pickle.dump(text_embeddings, fp)
            # chunk_id += 1