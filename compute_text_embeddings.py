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
text_embeddings_dict = {}
save_path = "/mnt/wangxiaofa/robot_dataset/t5_embeddings_pretrain.pkl"
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
    with open(val2root_json_path, "r") as f:
        name2path_dict = json.load(f)
    
    for d_name, ratio in name2path_dict.items():
        d_path = name2path_dict[d_name]
        data_path = os.path.join(data_root, d_path)
        if os.path.exists(data_path):
            task_path = os.path.join(data_path, "meta", "tasks.jsonl")
            tasks = []
            with open(task_path, "r") as f:
                for line in f:
                    d_dict = json.loads(line)
                    tasks.append((d_dict["task_index"], d_dict["task"]))
            text_embeddings_dict = {}
            text_embeddings = []
            for i in tqdm(range(len(tasks))):
                t_id, prompts = tasks[i]
                with torch.no_grad():
                    encoded_text = encode_t5_text_embeddings(text_encoder, tokenizer, prompts, 
                                                            max_length=max_length, device=device)
                encoded_text = encoded_text.cpu().numpy().astype(np.float16)
                # text_embeddings[start:end] = encoded_text
                text_embeddings.append(encoded_text)
                # if len(text_embeddings) > 10:
                #     break
                # print(encoded_text.shape)
            text_embeddings_dict[d_name] = text_embeddings
            break
    
    with open(save_path, "wb") as fp:
        pickle.dump(encoded_text, fp)
