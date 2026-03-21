# ------------------------- minimind config ------------------------- #
from typing import Any

from transformers import PretrainedConfig


class MinimindConfig(PretrainedConfig):
    # 模型配置类，继承自 Hugging Face 的 PretrainedConfig 类
    # 
    model_type = "minimind"

    def __init__(
        self,
        dropout: float = 0.0,                       # 正则化的dropout率
        bos_token_id: int = 1,                      # 序列开始标记的ID
        eos_token_id: int = 2,                      # 序列结束标记的ID
        hidden_act: str = "silu",                   # 隐藏层的激活函数
        hidden_size: int = 512,                     # 隐藏层的大小
        intermediate_size: int = None,              # 前馈网络中中间层的大小
        max_position_embeddings: int = 32768,       # 模型能处理的最大序列长度
        num_attention_heads: int = 8,               # 注意力头的数量
        num_hidden_layers: int = 8,                 # 隐藏层的数量
        num_key_value_heads: int = 2,               # 分组查询注意力的键值头数量
        vocab_size: int = 6400,                     # 词汇表大小
        rms_norm_eps: float = 1e-05,                # RMS归一化的epsilon值
        rope_theta: int = 1000000,                  # RoPE位置编码的theta值
        inference_rope_scaling: bool = False,       # 是否在推理时使用RoPE缩放
        flash_attention: bool = True,               # 是否使用flash attention以加快计算
        ############ MoE ############
        use_moe: bool = False,                      # 是否使用专家混合(Mixture of Experts)架构
        num_experts_per_tok: int = 2,               # 每个token使用的专家数量
        n_routed_experts: int = 4,                  # 路由选择的专家数量
        n_shared_experts: int = 1,                  # 共享专家的数量
        scoring_func: str = "softmax",              # 专家评分函数
        aux_loss_alpha: float = 0.01,               # 辅助损失的权重
        seq_aux: bool = True,                       # 是否使用序列辅助损失
        norm_topk_prob: bool = True,                # 是否归一化top-k概率
        **kwargs,
    ):
        
        # 调用父类构造函数
        super().__init__(**kwargs)

        # 设置基本模型参数
        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.inference_rope_scaling = inference_rope_scaling
        self.flash_attention = flash_attention
        
        # 设置MoE参数
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.seq_aux = seq_aux
        self.norm_topk_prob = norm_topk_prob
        self.aux_loss_alpha = aux_loss_alpha
        self.scoring_func = scoring_func

        # 如果启用RoPE缩放，则配置相关参数
        self.rope_scaling = (
            {
                "beta_fast": 32,  # YARN缩放的快速beta参数
                "beta_slow": 1,   # YARN缩放的慢速beta参数
                "factor": 16,     # 缩放因子
                "original_max_position_embeddings": 2048,  # 原始最大位置嵌入
                "attention_factor": 1.0,    # 注意力因子
                "type": "yarn",             # RoPE缩放类型
            }
            if self.inference_rope_scaling
            else None
        )

# 导入依赖
import torch
import torch.nn as nn


# 一、RMSNorm层归一化模型
class MinimindModel(nn.Module):

    # __init__初始化
    def __init__(self, dim:int, eps:float = 1e-05):
        super().__init__()
        self.dim = dim  
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    # _norm归一化层，使用RMSNorm进行归一化
    def _norm(self, x):     # 对输入张量x平方，乘上：（其的平均值去倒数，再加eps，最后开放并取倒数）
        return x*torch.rsqrt(x.pow(2).mean(-1, keepdim = True) + self.eps)

    # forward
    def forward(self, x):   
        return self.weight * self._norm(x.float()).type_as(x)


# 二、RoPE将位置编码转换为旋转矩阵
# 1、预先计算旋转位置编码所需的 Cos 和 Sin 矩阵
def precompute_freqs_cis(dim: int,                               # 位置编码的维度
                        end: int = int(32 * 1024),              # 可能的序列最大长度，上下文长度
                        rope_base: float = 1e6,                 # 位置编码频率底数
                        rope_scaling: Optional[dict] = None     # 长上下文缩放方式：不缩放
):
    pass
    # 1.初始化RoPE频率



