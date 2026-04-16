"""
测试目标：
- 对 `model.generate(...)` 做最小冒烟验证。
- 确认模型在启用生成接口后至少能连续生成若干 token。

预期结果：
- `generate()` 不应在前几步崩溃。
- 返回结果长度应大于原始 prompt 长度。

测试步骤：
1. 构造一个小尺寸 MiniMind 模型。
2. 准备一段最小输入及 attention mask。
3. 调用 `generate()` 生成少量新 token。
4. 检查返回张量形状。
"""
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from test._shared import build_small_model


def main():
    """执行最小生成冒烟测试。"""
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
