"""
测试目标：
- 独立验证 `RMSNorm` 的输出形状、dtype 和数值稳定性。

预期结果：
- 输出形状与输入一致。
- 输出 dtype 与输入 dtype 保持一致。
- 输出张量中不应出现 NaN 或 Inf。

测试步骤：
1. 构造 RMSNorm 层。
2. 输入一批随机张量。
3. 检查输出形状、dtype 与数值有效性。
"""
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.model_minimind import RMSNorm
from test._shared import assert_all_finite


def main():
    """执行 RMSNorm 模块测试。"""
    torch.manual_seed(42)
    layer = RMSNorm(128).eval()
    hidden_states = torch.randn(2, 5, 128, dtype=torch.float32)

    with torch.no_grad():
        output = layer(hidden_states)

    assert output.shape == hidden_states.shape
    assert output.dtype == hidden_states.dtype
    assert_all_finite(output)

    print("test_rmsnorm: PASS")


if __name__ == "__main__":
    main()
