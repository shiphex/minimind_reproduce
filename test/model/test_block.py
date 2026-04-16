"""
测试目标：
- 独立验证 `MiniMindBlock` 的残差链路与缓存返回结构。

预期结果：
- Block 前向输出形状与输入隐藏状态一致。
- `present_key_value` 为 `(k, v)`，并支持增量场景继续增长。

测试步骤：
1. 构造一个小尺寸 Block。
2. 跑一轮完整输入，检查输出和缓存。
3. 再跑一轮单 token 增量输入，检查缓存长度增长。
"""
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.model_minimind import MiniMindBlock
from test._shared import build_position_embeddings, build_small_config, assert_all_finite


def main():
    """执行 Transformer Block 测试。"""
    torch.manual_seed(42)
    config = build_small_config()
    block = MiniMindBlock(layer_id=0, config=config).eval()

    hidden_states = torch.randn(2, 5, config.hidden_size)
    attention_mask = torch.ones(2, 5, dtype=torch.long)
    position_embeddings = build_position_embeddings(config, seq_len=5)

    # 首轮前向：验证输出形状和缓存结构。
    with torch.no_grad():
        output, present = block(
            hidden_states,
            position_embeddings,
            use_cache=True,
            attention_mask=attention_mask,
        )

    assert output.shape == hidden_states.shape
    assert isinstance(present, tuple) and len(present) == 2
    assert present[0].shape[1] == 5
    assert present[1].shape[1] == 5
    assert_all_finite(output)

    # 第二轮增量前向：验证缓存长度增加。
    next_hidden = torch.randn(2, 1, config.hidden_size)
    next_position_embeddings = build_position_embeddings(config, seq_len=1, start_pos=5)
    with torch.no_grad():
        output2, present2 = block(
            next_hidden,
            next_position_embeddings,
            past_key_value=present,
            use_cache=True,
            attention_mask=attention_mask,
        )

    assert output2.shape == (2, 1, config.hidden_size)
    assert present2[0].shape[1] == 6
    assert present2[1].shape[1] == 6

    print("test_block: PASS")


if __name__ == "__main__":
    main()
