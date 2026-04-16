"""
测试目标：
- 回归验证 `PretrainDataset` 的 BOS/EOS/PAD 处理与 label 掩码逻辑。

预期结果：
- `input_ids` 与 `labels` 长度都等于 `max_length`。
- 第一个 token 为 BOS，最后一个非 PAD token 为 EOS。
- PAD 位置在 `labels` 中被置为 `-100`。

测试步骤：
1. 加载本地 tokenizer。
2. 用最小样本伪造一个数据集条目。
3. 取出一条样本并检查输入输出约束。
"""
from pathlib import Path
import sys
import tempfile

import torch
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset.lm_dataset import PretrainDataset
import dataset.lm_dataset as lm_dataset
from trainer.trainer_utils import DEFAULT_MODEL_DIR


def main():
    """执行预训练数据集样本构造测试。"""
    tokenizer = AutoTokenizer.from_pretrained(str(DEFAULT_MODEL_DIR))

    # 用内存假数据替代真实数据集加载，确保测试聚焦在编码逻辑本身。
    original_load_dataset = lm_dataset.load_dataset
    lm_dataset.load_dataset = lambda *args, **kwargs: [{"text": "你好，MiniMind"}]
    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            data_path = Path(tmp_dir) / "sample.jsonl"
            data_path.write_text('{"text": "你好，MiniMind"}\n', encoding="utf-8")

            dataset = PretrainDataset(str(data_path), tokenizer, max_length=12)
            input_ids, labels = dataset[0]
    finally:
        lm_dataset.load_dataset = original_load_dataset

    assert isinstance(input_ids, torch.Tensor)
    assert isinstance(labels, torch.Tensor)
    assert input_ids.shape == (12,)
    assert labels.shape == (12,)
    assert input_ids[0].item() == tokenizer.bos_token_id

    # 找到最后一个非 PAD 位置，确认其为 EOS。
    non_pad_positions = (input_ids != tokenizer.pad_token_id).nonzero(as_tuple=True)[0]
    assert len(non_pad_positions) > 0
    assert input_ids[non_pad_positions[-1]].item() == tokenizer.eos_token_id

    # PAD 位置在 labels 中必须被屏蔽为 -100。
    pad_mask = input_ids == tokenizer.pad_token_id
    assert torch.all(labels[pad_mask] == -100)
    assert torch.equal(labels[~pad_mask], input_ids[~pad_mask])

    print("test_pretrain_dataset: PASS")


if __name__ == "__main__":
    main()
