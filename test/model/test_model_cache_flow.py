"""
测试目标：
- 验证 `MiniMindForCausalLM.forward(..., use_cache=True)` 会返回可复用的 `past_key_values`。
- 验证增量前向时，新缓存长度会在上一轮基础上增长。

预期结果：
- 第一轮前向能返回与层数一致的缓存列表。
- 第二轮只输入最后一个 token 时，缓存长度会增加 1，且 logits 形状正确。

测试步骤：
1. 构造一个小尺寸 MiniMind 模型。
2. 先用完整输入做一次带 cache 的前向。
3. 再只输入最后一个 token，并传入第一轮 cache。
4. 检查缓存结构、长度增长和 logits 形状。
"""
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from test._shared import build_small_model


def main():
    """执行 KV cache 数据流测试。"""
    torch.manual_seed(42)
    model = build_small_model()

    input_ids = torch.randint(0, model.config.vocab_size, (1, 7))
    attention_mask = torch.ones_like(input_ids)

    # 第一轮前向：验证会返回完整层级的缓存。
    with torch.no_grad():
        out1 = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)

    assert out1.past_key_values is not None
    assert isinstance(out1.past_key_values, list)
    assert len(out1.past_key_values) == model.config.num_hidden_layers

    first_layer_k, first_layer_v = out1.past_key_values[0]
    assert first_layer_k.shape[:2] == (1, 7)
    assert first_layer_v.shape[:2] == (1, 7)

    # 第二轮前向：只输入最后一个 token，验证缓存长度会增长 1。
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
