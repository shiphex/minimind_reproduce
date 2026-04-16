"""
测试目标：
- 验证 `MiniMindForCausalLM.forward` 在训练与缓存场景下的核心输出。

预期结果：
- 同时提供 `loss`、`logits`、`past_key_values` 和 `aux_loss`。
- `loss` 为标量，`logits` 形状正确，缓存层数与模型层数一致。

测试步骤：
1. 构造一个小尺寸 MiniMind 模型。
2. 准备输入与 labels。
3. 开启 `use_cache=True` 做前向。
4. 检查返回对象中的各个关键字段。
"""
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from test._shared import build_small_model


def main():
    """执行 CausalLM forward 测试。"""
    torch.manual_seed(42)
    model = build_small_model()

    input_ids = torch.randint(0, model.config.vocab_size, (2, 6))
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()

    with torch.no_grad():
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=True,
        )

    assert output.loss is not None
    assert output.loss.ndim == 0
    assert output.logits.shape == (2, 6, model.config.vocab_size)
    assert output.past_key_values is not None
    assert len(output.past_key_values) == model.config.num_hidden_layers
    assert hasattr(output, "aux_loss")
    assert output.aux_loss.ndim == 0

    print("test_model_forward: PASS")


if __name__ == "__main__":
    main()
