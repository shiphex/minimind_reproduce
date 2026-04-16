from pathlib import Path
import sys
import tempfile

import torch
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset.lm_dataset import PretrainDataset
import dataset.lm_dataset as lm_dataset
from trainer.trainer_utils import DEFAULT_MODEL_DIR


def main():
    tokenizer = AutoTokenizer.from_pretrained(str(DEFAULT_MODEL_DIR))

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

    non_pad_positions = (input_ids != tokenizer.pad_token_id).nonzero(as_tuple=True)[0]
    assert len(non_pad_positions) > 0
    assert input_ids[non_pad_positions[-1]].item() == tokenizer.eos_token_id

    pad_mask = input_ids == tokenizer.pad_token_id
    assert torch.all(labels[pad_mask] == -100)
    assert torch.equal(labels[~pad_mask], input_ids[~pad_mask])

    print("test_pretrain_dataset: PASS")


if __name__ == "__main__":
    main()
