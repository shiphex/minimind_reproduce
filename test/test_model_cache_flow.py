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

    assert out1.past_key_values is not None
    assert isinstance(out1.past_key_values, list)
    assert len(out1.past_key_values) == model.config.num_hidden_layers

    first_layer_k, first_layer_v = out1.past_key_values[0]
    assert first_layer_k.shape[:2] == (1, 7)
    assert first_layer_v.shape[:2] == (1, 7)

    next_ids = input_ids[:, -1:]
    with torch.no_grad():
        out2 = model(
            input_ids=next_ids,
            attention_mask=attention_mask,
            past_key_values=out1.past_key_values,
            use_cache=True,
        )

    second_layer_k, second_layer_v = out2.past_key_values[0]
    assert second_layer_k.shape[1] == first_layer_k.shape[1] + 1
    assert second_layer_v.shape[1] == first_layer_v.shape[1] + 1
    assert out2.logits.shape == (1, 1, model.config.vocab_size)

    print("test_model_cache_flow: PASS")


if __name__ == "__main__":
    main()
