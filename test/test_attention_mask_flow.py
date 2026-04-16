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

    input_ids = torch.randint(0, model.config.vocab_size, (1, 7))
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        out1 = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)

    next_ids = input_ids[:, -1:]
    short_mask = attention_mask
    long_mask = torch.ones((1, 12), dtype=attention_mask.dtype)

    with torch.no_grad():
        out_short = model(
            input_ids=next_ids,
            attention_mask=short_mask,
            past_key_values=out1.past_key_values,
            use_cache=True,
        )
        out_long = model(
            input_ids=next_ids,
            attention_mask=long_mask,
            past_key_values=out1.past_key_values,
            use_cache=True,
        )

    assert out_short.logits.shape == (1, 1, model.config.vocab_size)
    assert out_long.logits.shape == (1, 1, model.config.vocab_size)
    assert out_short.past_key_values[0][0].shape[1] == 8
    assert out_long.past_key_values[0][0].shape[1] == 8

    print("test_attention_mask_flow: PASS")


if __name__ == "__main__":
    main()
