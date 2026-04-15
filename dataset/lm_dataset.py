from torch.utils.data import Dataset
import torch
import json
import os
import random
from datasets import load_dataset, Features, Sequence, Value


# 自回归预测
class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length = 512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = load_dataset('json', data_files=data_path, split = 'train')

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        sample = self.samples[index]
        # 1. tokenizer原始文本其前后各留下1个token位置给 BOS/EOS
        tokens = self.tokenizer(str(sample['text']), 
                                add_special_tokens = False, 
                                max_length = self.max_length -2, 
                                truncation = True)
        # 2. 拼接 BOS + token文本 + EOS 组成完整训练序列
        tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]
        # 3. 使用 PAD 补齐文本长度到统一的 max_length
        input_ids = tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(tokens))
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        # 4. 复制一份 input_id 作为训练标签，避免修改原张量
            # labels 与 input_ids 完全相同，但 PAD 位置置 -100，
            # CrossEntropyLoss 会自动忽略 -100，不计入 loss
        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return input_ids, labels


# 
