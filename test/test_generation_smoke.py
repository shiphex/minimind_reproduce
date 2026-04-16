from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM


def build_small_model():
    config = MiniMindConfig(
        hidden_size=128,
        num_hidden_layers=2,
        use_moe=False,
        vocab_size=128,
        flash_attn=False,
    )
    return MiniMindForCausalLM(config).eval()


def main():
    torch.manual_seed(42)
    model = build_small_model()

    input_ids = torch.tensor([[1, 10, 11, 12, 13]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        generated_ids = model.generate(
            inputs=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=4,
            do_sample=False,
            pad_token_id=0,
            eos_token_id=2,
        )

    assert generated_ids.shape[0] == 1
    assert generated_ids.shape[1] >= input_ids.shape[1] + 1

    print("test_generation_smoke: PASS")


if __name__ == "__main__":
    main()
