"""
测试目标：
- 独立验证 `Attention` 模块的输入输出维度与缓存结构。
- 覆盖首轮前向和带历史 KV cache 的增量前向。

预期结果：
- 首轮输出形状为 `[batch, seq_len, hidden_size]`。
- 返回的缓存应为 `(k, v)` 二元组，且第二轮长度正常增长。

测试步骤：
1. 构造 Attention 模块与对应位置编码。
2. 跑一轮完整输入，检查输出与缓存形状。
3. 再跑一轮单 token 增量输入，检查缓存长度增长。
"""
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.model_minimind import Attention
from test._shared import build_position_embeddings, build_small_config, assert_all_finite


def main():
    """执行 Attention 模块测试。"""
    torch.manual_seed(42)
    config = build_small_config()
    attention = Attention(config).eval()

    hidden_states = torch.randn(2, 5, config.hidden_size)
    attention_mask = torch.ones(2, 5, dtype=torch.long)
    position_embeddings = build_position_embeddings(config, seq_len=5)

    # 首轮前向：检查输出形状和缓存结构。
    with torch.no_grad():
        output, past = attention(
            hidden_states,
            position_embeddings,
            use_cache=True,
            attention_mask=attention_mask,
        )

    assert output.shape == (2, 5, config.hidden_size)
    assert isinstance(past, tuple) and len(past) == 2
    assert past[0].shape == (2, 5, config.num_key_value_heads, config.head_dim)
    assert past[1].shape == (2, 5, config.num_key_value_heads, config.head_dim)
    assert_all_finite(output)

    # 增量前向：只输入一个 token，验证缓存长度增加。
    next_hidden = torch.randn(2, 1, config.hidden_size)
    next_position_embeddings = build_position_embeddings(config, seq_len=1, start_pos=5)
    with torch.no_grad():
        output2, past2 = attention(
            next_hidden,
            next_position_embeddings,
            past_key_value=past,
            use_cache=True,
            attention_mask=attention_mask,
        )

    assert output2.shape == (2, 1, config.hidden_size)
    assert past2[0].shape[1] == 6
    assert past2[1].shape[1] == 6

    print("test_attention: PASS")


if __name__ == "__main__":
    main()
