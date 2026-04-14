from torch.utils.data import Dataset
import torch
import json
import os
import random
from datasets import load_dataset, Features, Sequence, Value


# 
class PretrainDataset(Dataset):
    pass
    # 1. tokenizer原始文本其前后各留下1个token位置给 BOS/EOS
    # 2. 拼接 BOS + token文本 + EOS 组成完整训练序列
    # 3. 使用 PAD 补齐文本长度到统一的 max_length
    # 4. 复制一份 input_id 作为训练标签，避免修改原张量
        # labels 与 input_ids 完全相同，但 PAD 位置置 -100，
        # CrossEntropyLoss 会自动忽略 -100，不计入 loss
    # 5. 返回 attention_mask，使 attention 层能屏蔽 padding token
