"""
测试目标：
- 验证增量生成阶段 `attention_mask` 与注意力分数的长度可以对齐。
- 覆盖 mask 偏短与偏长两种防御性分支。

预期结果：
- 使用短 mask 和长 mask 时，前向都不应报错。
- 两种场景下返回的 logits 形状正确，缓存长度都能正常增长。

测试步骤：
1. 构造一个小尺寸 MiniMind 模型。
2. 先跑一轮完整输入，拿到历史缓存。
3. 再分别传入短 mask 和长 mask 做单 token 增量前向。
4. 检查 logits 与缓存长度。
"""
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from test._shared import build_small_model


def main():
    """执行 attention mask 长度对齐测试。"""
    torch.manual_seed(42)
    model = build_small_model()

    input_ids = torch.randint(0, model.config.vocab_size, (1, 7))
    attention_mask = torch.ones_like(input_ids)

    # 先建立一轮历史缓存，模拟真实生成链路。
    with torch.no_grad():
        out1 = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)

    next_ids = input_ids[:, -1:]
    short_mask = attention_mask
    long_mask = torch.ones((1, 12), dtype=attention_mask.dtype)

    # 分别验证 mask 偏短与偏长时都能被内部逻辑安全处理。
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
