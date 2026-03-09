import os
import json
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class ChickenDiseaseDataset(Dataset):
    def __init__(self, jsonl_path, root_dir, processor):
        self.root_dir = root_dir
        self.processor = processor
        self.data = []
        
        self.image_index = {}
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
                    self.image_index[file] = os.path.join(root, file)

        with open(jsonl_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for line in lines:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                img_path = item.get('img_path') or item.get('image')
                if img_path:
                    self.data.append(item)
            except:
                continue
        
        print(f"Loaded {len(self.data)} items from {jsonl_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        max_retry = 5
        
        for attempt in range(max_retry):
            try:
                item = self.data[idx]
                img_path = item.get('img_path') or item.get('image')
                img_filename = os.path.basename(img_path)
                
                full_path = self.image_index.get(img_filename)
                if not full_path or not os.path.exists(full_path):
                    raise FileNotFoundError(f"Image not found: {img_filename}")
                
                image = Image.open(full_path).convert("RGB")

                conversation = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": item['question']},
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": item['answer']}],
                    },
                ]

                text = self.processor.apply_chat_template(
                    conversation, tokenize=False, add_generation_prompt=False
                )
                inputs = self.processor(
                    text=[text], images=[image], padding=True, return_tensors="pt"
                )

                labels = inputs["input_ids"][0].clone()
                
                assistant_tokens = self.processor.tokenizer.encode(
                    "<|im_start|>assistant", add_special_tokens=False
                )
                if len(assistant_tokens) > 0:
                    seq_len = len(assistant_tokens)
                    for i in range(len(labels) - seq_len, -1, -1):
                        if torch.all(labels[i:i+seq_len] == torch.tensor(assistant_tokens)):
                            labels[:i+seq_len] = -100
                            break

                eos_token_id = self.processor.tokenizer.eos_token_id
                if eos_token_id:
                    eos_pos = (labels == eos_token_id).nonzero(as_tuple=True)[0]
                    if len(eos_pos) > 0:
                        eos_idx = eos_pos[0].item()
                        if eos_idx + 1 < len(labels):
                            labels[eos_idx+1:] = -100

                return {
                    "input_ids": inputs["input_ids"][0],
                    "attention_mask": inputs["attention_mask"][0],
                    "pixel_values": inputs["pixel_values"],
                    "image_grid_thw": inputs["image_grid_thw"],
                    "labels": labels,
                }
            
            except Exception as e:
                if attempt < max_retry - 1:
                    idx = random.randint(0, len(self.data) - 1)
                    continue
                else:
                    print(f"Failed after {max_retry} retries: {e}")
                    return None
