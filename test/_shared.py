"""
测试共享工具。

用途：
- 统一处理项目根目录注入，保证分层后的脚本仍可直接运行。
- 提供最小模型、最小配置和位置编码切片工具，减少测试脚本重复代码。
"""
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.model_minimind import (  # noqa: E402
    MiniMindConfig,
    MiniMindForCausalLM,
    precompute_freqs_cis,
)


def build_small_config():
    """构造一个足够小、适合单元测试的模型配置。"""
    return MiniMindConfig(
        hidden_size=128,
        num_hidden_layers=2,
        use_moe=False,
        vocab_size=128,
        flash_attn=False,
    )


def build_small_model():
    """构造一个 eval 模式下的小模型，便于快速做前向和生成测试。"""
    return MiniMindForCausalLM(build_small_config()).eval()


def build_position_embeddings(config, seq_len, start_pos=0):
    """按给定长度切出当前测试所需的 RoPE 位置编码。"""
    freqs_cos, freqs_sin = precompute_freqs_cis(
        dim=config.head_dim,
        end=config.max_position_embeddings,
        rope_base=config.rope_theta,
        rope_scaling=config.rope_scaling,
    )
    return (
        freqs_cos[start_pos : start_pos + seq_len],
        freqs_sin[start_pos : start_pos + seq_len],
    )


def assert_all_finite(tensor):
    """断言张量中不存在 NaN 或 Inf。"""
    assert torch.isfinite(tensor).all(), "Tensor contains NaN or Inf values."
