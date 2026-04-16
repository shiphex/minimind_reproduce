"""
测试目标：
- 独立验证 `FeedForward` 模块的输入输出维度与数值稳定性。

预期结果：
- 输出形状与输入形状一致。
- 前向不应报错，且输出中不应出现 NaN 或 Inf。

测试步骤：
1. 构造 FeedForward 层。
2. 输入一批随机隐藏状态。
3. 检查输出形状和数值有效性。
"""
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.model_minimind import FeedForward
from test._shared import build_small_config, assert_all_finite


def main():
    """执行 FeedForward 模块测试。"""
    torch.manual_seed(42)
    config = build_small_config()
    layer = FeedForward(config).eval()
    hidden_states = torch.randn(2, 5, config.hidden_size)

    with torch.no_grad():
        output = layer(hidden_states)

    assert output.shape == hidden_states.shape
    assert_all_finite(output)

    print("test_ffn: PASS")


if __name__ == "__main__":
    main()
