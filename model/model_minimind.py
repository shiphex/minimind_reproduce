# 导入依赖
import math, torch, torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_outputs import MoeCausalLMOutputWithPast


# ------------------------- minimind config ------------------------- #
from typing import Any
from transformers import PretrainedConfig


# 模型配置类，继承自 Hugging Face 的 PretrainedConfig 类
class MiniMindConfig(PretrainedConfig):
    model_type = "minimind"
    def __init__(self, 
                 hidden_size = 768,     # 隐藏层的大小
                 num_hidden_layers = 8, # 隐藏层的数量
                 use_moe = False,       # 是否使用专家混合(Mixture of Experts)架构
                 **kwargs):
        super().__init__(**kwargs)

        # 模型基础参数：
            # hidden_size               隐藏层的大小            = d_model = dim = 单个token的向量长度
            # num_hidden_layers         隐藏层的数量            Transformer Layers 层的串联个数
            # use_moe                   是否使用专家混合(Mixture of Experts)架构
            # dropout                   正则化的dropout率
            # vocab_size                词汇表大小
            # bos_token_id              序列开始标记的ID
            # eos_token_id              序列结束标记的ID
            # flash_attn                是否使用flash attention以加快计算 (需硬件支持)
            # num_attention_heads       注意力头的数量
            # num_key_value_heads       分组查询注意力的键值头数量
            # head_dim                  注意力头的维度
            # hidden_act                隐藏层的激活函数
            # intermediate_size         前馈网络中中间层的大小
            # max_position_embeddings   模型能处理的最大序列长度
            # rms_norm_eps              RMS归一化的epsilon值
            # rope_theta                RoPE位置编码的theta值，置编码频率底数
            # inference_rope_scaling    是否在推理时使用RoPE缩放

        self.hidden_size            = hidden_size
        self.num_hidden_layers      = num_hidden_layers
        self.use_moe                = use_moe
        self.dropout                = kwargs.get("dropout", 0.0)
        self.vocab_size             = kwargs.get("vocab_size", 6400)
        self.bos_token_id           = kwargs.get("bos_token_id", 1)
        self.eos_token_id           = kwargs.get("eos_token_id", 2)
        self.flash_attn             = kwargs.get("flash_attn", True)
        self.num_attention_heads    = kwargs.get("num_attention_heads", 8)
        self.num_key_value_heads    = kwargs.get("num_key_value_heads", 4)
        self.head_dim               = kwargs.get("head_dim", self.hidden_size // self.num_attention_heads)
        self.hidden_act             = kwargs.get("hidden_act", 'silu')
        self.intermediate_size      = kwargs.get("intermediate_size", math.ceil(hidden_size * math.pi / 64) * 64)
        self.max_position_embeddings = kwargs.get("max_position_embeddings", 32768)
        self.rms_norm_eps           = kwargs.get("rms_norm_eps", 1e-6)
        self.rope_theta             = kwargs.get("rope_theta", 1e6)
        self.inference_rope_scaling = kwargs.get("inference_rope_scaling", False)

        # 如果启用RoPE缩放，则配置相关参数
        self.rope_scaling = {                       # RoPE缩放配置参数：
            "beta_fast": 32,                            # 快速RoPE缩放的beta值(高频边界)
            "beta_slow": 1,                             # 慢速RoPE缩放的beta值(低频边界)    
            "factor": 16,                               # RoPE缩放的因子
            "original_max_position_embeddings": 2048,   # 原始RoPE缩放的最大序列长度
            "attention_factor": 1.0,                    # 注意力缩放因子
            "type": "yarn"                              # RoPE缩放的类型("yarn"或"linear")
        } if self.inference_rope_scaling else None

        # MOE (混合专家模型) 专属配置区域
        ### MoE specific configs (ignored if use_moe = False)
        # MOE配置参数：
            # num_experts               专家数量
            # num_experts_per_tok       Top-K：每个 Token 在推理时激活的专家数量 (通常远小于总专家数)
            # moe_intermediate_size     MoE 前馈网络中中间层的大小
            # norm_topk_prob            是否对选出的 Top-K 专家的权重进行归一化 (使其和为1)
            # router_aux_loss_coef      均衡损失权重
        self.num_experts            = kwargs.get("num_experts", 4)
        self.num_experts_per_tok    = kwargs.get("num_experts_per_tok", 1)
        self.moe_intermediate_size  = kwargs.get("moe_intermediate_size", self.intermediate_size)
        self.norm_topk_prob         = kwargs.get("norm_topk_prob", True)
        self.router_aux_loss_coef   = kwargs.get("router_aux_loss_coef", 5e-4)
        # MoE for shared experts 配置参数：
            # num_routed_experts        路由选择的专家数量
            # num_shared_experts        共享专家的数量
            # scoring_func              评分函数，默认为'softmax'
            # aux_loss_alpha            辅助损失 alpha 权重，默认为0.1
            # seq_aux                   是否在序列级别上计算辅助损失
        self.num_routed_experts = kwargs.get("num_routed_experts", self.num_experts)
        self.num_shared_experts = kwargs.get("num_shared_experts", int(1))
        self.scoring_func = kwargs.get("scoring_func", "softmax")
        self.aux_loss_alpha = kwargs.get("aux_loss_alpha", float(0.1))
        self.seq_aux = kwargs.get("seq_aux", bool(True))

# 废弃的config
'''
# class MiniMindConfig(PretrainedConfig):
#     # 模型配置类，继承自 Hugging Face 的 PretrainedConfig 类
#     # 
#     model_type = "minimind"
# 
#     def __init__(
#         self,
#         dropout: float = 0.0,                       # 正则化的dropout率
#         bos_token_id: int = 1,                      # 序列开始标记的ID
#         eos_token_id: int = 2,                      # 序列结束标记的ID
#         hidden_act: str = "silu",                   # 隐藏层的激活函数
#         hidden_size: int = 512,                     # 隐藏层的大小
#         intermediate_size: int = None,              # 前馈网络中中间层的大小
#         max_position_embeddings: int = 32768,       # 模型能处理的最大序列长度
#         num_attention_heads: int = 8,               # 注意力头的数量
#         num_hidden_layers: int = 8,                 # 隐藏层的数量
#         num_key_value_heads: int = 2,               # 分组查询注意力的键值头数量
#         vocab_size: int = 6400,                     # 词汇表大小
#         rms_norm_eps: float = 1e-05,                # RMS归一化的epsilon值
#         rope_theta: int = 1000000,                  # RoPE位置编码的theta值
#         inference_rope_scaling: bool = False,       # 是否在推理时使用RoPE缩放
#         flash_attention: bool = True,               # 是否使用flash attention以加快计算
#         ############ MoE ############
#         use_moe: bool = False,                      # 是否使用专家混合(Mixture of Experts)架构
#         num_experts_per_tok: int = 2,               # 每个token使用的专家数量
#         num_routed_experts: int = 4,                  # 路由选择的专家数量
#         num_shared_experts: int = 1,                  # 共享专家的数量
#         scoring_func: str = "softmax",              # 专家评分函数
#         aux_loss_alpha: float = 0.01,               # 辅助损失的权重
#         seq_aux: bool = True,                       # 是否使用序列辅助损失
#         norm_topk_prob: bool = True,                # 是否归一化top-k概率
#         **kwargs,
#     ):
#         
#         # 调用父类构造函数
#         super().__init__(**kwargs)
# 
#         # 设置基本模型参数
#         self.dropout = dropout
#         self.bos_token_id = bos_token_id
#         self.eos_token_id = eos_token_id
#         self.hidden_act = hidden_act
#         self.hidden_size = hidden_size
#         self.intermediate_size = intermediate_size
#         self.max_position_embeddings = max_position_embeddings
#         self.num_attention_heads = num_attention_heads
#         self.num_hidden_layers = num_hidden_layers
#         self.num_key_value_heads = num_key_value_heads
#         self.vocab_size = vocab_size
#         self.rms_norm_eps = rms_norm_eps
#         self.rope_theta = rope_theta
#         self.inference_rope_scaling = inference_rope_scaling
#         self.flash_attention = flash_attention
#         
#         # 设置MoE参数
#         self.use_moe = use_moe
#         self.num_experts_per_tok = num_experts_per_tok
#         self.num_routed_experts = num_routed_experts
#         self.num_shared_experts = num_shared_experts
#         self.seq_aux = seq_aux
#         self.norm_topk_prob = norm_topk_prob
#         self.aux_loss_alpha = aux_loss_alpha
#         self.scoring_func = scoring_func
# 
#         # 如果启用RoPE缩放，则配置相关参数
#         self.rope_scaling = (
#             {
#                 "beta_fast": 32,  # YARN缩放的快速beta参数
#                 "beta_slow": 1,   # YARN缩放的慢速beta参数
#                 "factor": 16,     # 缩放因子
#                 "original_max_position_embeddings": 2048,  # 原始最大位置嵌入
#                 "attention_factor": 1.0,    # 注意力因子
#                 "type": "yarn",             # RoPE缩放类型
#             }
#             if self.inference_rope_scaling
#             else None
#         )
'''


# --------------------------------- Function --------------------------------- #


# 一、RMSNorm层归一化模型
class RMSNorm(nn.Module):

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
def precompute_freqs_cis(dim: int,                              # 位置编码的维度
                        end: int = int(32 * 1024),              # 可能的序列最大长度，上下文长度
                        rope_base: float = 1e6,                 # 位置编码频率底数
                        rope_scaling: dict = None     # 长上下文缩放方式：不缩放
):
    
    # 1.初始化RoPE频率
    freqs = 1.0/(rope_base**((torch.arange(0, dim, 2))[:dim//2].float()/dim))   
    attn_factor = 1.0       # 温差缩放

    # 2.从配置字典中提取 YaRN 的超参数
    # YaRN 算法 (长文本外推逻辑)，如果启用RoPE缩放，则根据缩放类型调整频率
    if rope_scaling is not None:
        # 来源：《YaRN: Efficient Context Window Extension of Large Language Models》
        # https://arxiv.org/abs/2309.00071
        # orig_max：模型预训时最大训练长度，未设置时默认2048
        # factor：缩放因子，拓展倍数，未设置时默认16
        # beta_fast：高频边界，波长比例大于此值的维度不缩放
        # beta_slow：低频边界，波长比例小于此值的维度全量缩放
        # 在两边界之间：平滑插值缩放
        # attention_factor：注意力因子，未设置时默认1.0
        orig_max = rope_scaling.get("original_max_position_embeddings", 2048)    
        factor = rope_scaling.get("factor", 16)
        beta_fast = rope_scaling.get("beta_fast", 32.0)
        beta_slow = rope_scaling.get("beta_slow", 1.0)
        attn_factor = rope_scaling.get("attention_factor", 1.0)

        # 推理长度大于训练长度，使用缩放
        if end / orig_max > 1.0:
            # 3.计算每个频率的波长，并根据波长与边界的关系计算缩放因子
            # 计算不同长度的序列对应的不同维度的
            def inv_dim(b):
                return (dim * math.log(orig_max / (b * 2 * math.pi))) \
                       / (2 * math.log(rope_base))
            
            # 4. 计算高频区和低频区的维度切分点
            high = min(inv_dim(beta_slow), dim // 2 -1)
            low = max(inv_dim(beta_fast), 0)

            # 5.计算混合因子
            ramp = torch.clamp((torch.arange(dim//2, device = freqs.device).float() - low) \
                                / max(high - low, 0.001), 
                                0, 
                                1.0)

            # 6.频率融合公式，针对不同频率采用不同缩放方式
            freqs = freqs*(1 - ramp * (1 + 1/factor))

    # 7.根据可能的序列最大长度，生成位置索引
    t = torch.arange(end, device = freqs.device)

    # 8.求每个token、每个维度组的旋转角度θ的矩阵
    # 外积：生成每个token、每个维度组的旋转角度θ的矩阵，形状为(end, dim//2)
    freqs = torch.outer(t, freqs).float()

    # 9.计算对应的cos、sin，并引入注意力补偿系数
    # 拼接矩阵以适配rotate_half参数
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim = -1) * attn_factor
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim = -1) * attn_factor

    return freqs_sin, freqs_cos


def apply_rotary_pos_emb(q, k, cos, sin, position_ids = None, unsqueeze_dim = 1):
    # 参数说明：
    # q：查询向量，形状为 [batch_size, seq_len, num_heads, head_dim]
    # k：键值向量，形状为 [batch_size, seq_len, num_heads, head_dim]
    # cos: 预计算的余弦频率，形状为 [seq_len, head_dim]
    # sin：预计算的正弦频率，形状为 [seq_len, head_dim]
    # position_ids：位置索引，形状为 [batch_size, seq_len]
        # batch_size（B）：一次喂给模型多少句话 / 多少个样本
        # seq_len（L）：每个句子有多少个 token（词 / 字）
        # num_heads（H）：多头注意力有多少个头
        # head_dim（D）：每个注意力头的特征维度

    # 辅助函数：将向量切分为两半，并交换顺序、取负，[x1, x2, x3, x4] -> [-x3, -x4, x1, x2]
        # 标准原版 RoPE（相邻成对）:分组(x1, x2), (x3, x4)
        # MiniMind 对半切交叉版:分组(x1, x3), (x2, x4)
        # MiniMind版运算压力小，但高频 / 低频位置耦合混乱、大模型深度堆叠后上限略低
    def rotate_half(x):
        return torch.cat((-x[..., x.shape[-1]//2:], x[..., :x.shape[-1]//2]), dim = -1)

    # unsqueeze_dim：需要扩展的维度索引，默认为1
        # unsqueeze_dim: 0 表示在第 0 维度上扩展，将 cos/sin 扩展为 [1, seq_len, head_dim]
        # unsqueeze_dim: 1 表示在第 1 维度上扩展，将 cos/sin 扩展为 [seq_len, 1, head_dim]
        # unsqueeze_dim: 2 表示在第 2 维度上扩展，将 cos/sin 扩展为 [seq_len, head_dim, 1]
    # 广播机制：计算时自动将最左侧维度对齐，将 cos/sin 扩展为 [batch_size, seq_len, 1, head_dim]
    # 返回:
        # q_embed: 编码后的查询向量
        # k_embed: 编码后的键向量
    q_embed = (q * cos.unsqueeze(unsqueeze_dim) + (rotate_half(q)) * sin.unsqueeze(unsqueeze_dim))
    k_embed = (k * cos.unsqueeze(unsqueeze_dim) + (rotate_half(k)) * sin.unsqueeze(unsqueeze_dim))

    return q_embed, k_embed


# 把 key /value 的注意力头数量重复 n_rep 倍，让它和 query 的头数对齐，从而能正常做 scaled dot-product attention
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    # 重复键值张量的最后一个维度 n_rep 次
    # 参数：
        # x：键值张量，形状为 [batch_size, seq_len, num_heads, head_dim]
        # n_rep：重复倍数
    # 返回：
        # x_rep：重复后的键值张量，形状为 [batch_size, seq_len, num_heads * n_rep, head_dim]

    # 解包形状：批次、序列长度、KV 头数、每个头维度
    bs, slen, num_key_value_heads, head_dim = x.shape

    if n_rep == 1:
        return x
            # None 等价于 unsqueeze(-2)，插入一个维度用来重复 n_rep 次
            # reshape 把 num_key_value_heads 和 n_rep 合并
    return x[:, :, :, None, :]  \
            .expand(bs, slen, num_key_value_heads, n_rep, head_dim) \
            .reshape(bs, slen, num_key_value_heads * n_rep, head_dim)


class Attention(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.num_key_value_heads = config.num_attention_heads \
                                if config.num_key_value_heads is None \
                                else config.num_key_value_heads
        assert config.num_attention_heads % self.num_key_value_heads == 0, \
                f"num_attention_heads ({config.num_attention_heads}) must be divisible by num_key_value_heads ({self.num_key_value_heads})"
        self.n_local_heads = config.num_attention_heads
        self.n_local_kv_heads = config.num_key_value_heads
        self.n_rap = self.n_local_heads // self.n_local_kv_heads

        # 把 token 向量按注意力头切分/组合，实现多头注意力 + GQA 分组查询的核心投影逻辑
            # q_proj：把 token 映射成 “所有头的 Q 拼接”，一次性生成所有注意力头的 Q 向量的拼接结果
            # k_proj /v_proj：生成 K 和 V（可能用 GQA 分组策略）
            # o_proj：把所有头的输出拼回 hidden_size，把多头信息融合成一个统一的 token 向量
                    # 模型主干维度不会被破坏，可继续流动到下一层
        self.head_dim = config.head_dim
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)
        
        self.q_norm = RMSNorm(self.head_dim, eps = config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps = config.rms_norm_eps)

        # attention、RESNet 正则化
            # attn_dropout：作用于注意力权重矩阵（SoftMax 输出的 attention score）
            # resid_dropout：作用于注意力层 / FFN 层的残差输出
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout

        # 是否使用加速注意力计算
            # hasattr：查询PyTorch里，有没有内置 scaled_dot_product_attention 这个函数
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and config.flash_attn
        

    def forward(self, 
                x, 
                position_embeddings, 
                past_key_value = None, 
                use_cache = False, 
                attention_mask = None):
        bsz, seq_len, _ = x.shape   # x.shape = [batch_size, seq_len, hidden_size]

        # 把 token 映射成 “所有头的 Q 拼接”，一次性生成所有注意力头的 Q、K、V 向量的拼接结果
            # xq 形状：[batch_size, seq_len, num_heads, head_dim]
            # xk 形状：[batch_size, seq_len, num_heads_kv, head_dim]
            # xv 形状：[batch_size, seq_len, num_heads_kv, head_dim]
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        # 将生成的 Q、K、V 向量的拼接结果拆解到对应的注意力头中
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)

        # 对 Q、K 向量进行 RMSNorm 正则化(老版本minimind中此处无 RMSNorm 层)
        xq = self.q_norm(xq)
        xk = self.k_norm(xk)

        # YaRN 相对位置编码
        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)

        # 拼接缓存历史的 key/value
            # past_key_value[0]：历史的 key 张量
            # past_key_value[1]：历史的 value 张量
            # 维度与作用：
                # past_key_value: [batch_size, seq_len_old, num_heads_kv, head_dim]
                # 新 key 和 value: [batch_size, 1, num_heads_kv, head_dim]
                # 拼接后 key 和 value：[batch_size, seq_len_old + 1, num_heads_kv, head_dim]
            # past_kv: 决定本轮计算出的键值对 要不要存起来，给下一轮推理复用（也就是 KV Cache）
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim = 1)
            xv = torch.cat([past_key_value[1], xv], dim = 1)
        past_kv = (xk, xv) if use_cache else None

        # 为多头注意力做维度变换 + 分组注意力的 KV 重复
            # xq 交换前：[batch_size, seq_len, num_heads, head_dim]
            # xq 交换后：[batch_size, num_heads, seq_len, head_dim]
            # xk、xv 重复并交换前：[batch_size, seq_len, num_heads_kv, head_dim]
            # xk、xv 重复并交换后：[batch_size, num_heads, seq_len, head_dim]
            # 交换的好处：每个头要独立算 Q*K^T，互不干扰
                        # H(num_heads、num_heads_kv) 变成了批量维度
                        # 每个头是一个独立的小矩阵：[seq_len, head_dim]
                        # 可以一次性对所有头做矩阵乘法
                        # PyTorch 的批量矩阵乘法（bmm/matmul），天然支持前面的维度作为批量，只对最后两维做矩阵乘
                        # 只对最后两个维度做乘法，前面所有维度都当作批量并行处理
        xq, xk, xv = (xq.transpose(1, 2),   \
                      repeat_kv(xk, self.n_rap).transpose(1, 2),    \
                      repeat_kv(xv, self.n_rap).transpose(1, 2))
        
        # 满足条件就用 Flash Attention 加速计算注意力：
            # self.flash：      开启了 Flash Attention 优化
            # seq_len > 1：     序列长度大于 1 才用，单 token 没必要
            # past_key_value：  没有历史 KV 缓存，也就是第一次推理 / 训练前向
            # attention_mask：  没有给任何掩码，也就是全注意力
            # torch.all(attention_mask == 1)：所有位置都参与注意力计算，没有被 mask 掉的位置，全部位置都是可见的
        if self.flash and (seq_len > 1)     \
                      and (past_key_value is None)    \
                      and (attention_mask is None or torch.all(attention_mask == 1)):
            # 训练 / 第一次编码（无缓存）→ 可以用 Flash Attention
            # 逐字生成（有 past_kv）→ 必须手动写注意力循环，不能用这个接口
            # scaled_dot_product_attention：PyTorch 内置的 Flash Attention 函数
                # 接收完整 Q、完整 K、完整 V
                # 自己内部生成因果掩码（下三角掩码）
                # 一次性算完整个序列的注意力
            output = F.scaled_dot_product_attention(xq, xk, xv,     \
                                                    dropout_p = self.dropout if self.training else 0.0,     \
                                                    is_causal = True)
        else:
            # 计算注意力分数
                # scores 形状：[batch_size, num_heads, seq_len, seq_len_kv]
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
            # 因果掩码：对分数进行mask掩码处理，只保留下三角(含对角线)部分
                # 这时候的 scores 形状：[batch_size, num_heads, seq_len, seq_len_kv]，seq_len = 历史 KV 长度 + 本次新增长度
                # scores 最后一维只取最后 seq_len 个位置
                # 生成一个上三角矩阵，对角线以上为 -inf，对角线以下(含对角线)为 0
                # 这样在计算注意力时，上三角部分(不含对角线)的分数会被忽略，防止看到未来信息，只保留下三角部分(含对角线)
            scores[:, :, :, -seq_len:] += torch.full((seq_len, seq_len),     \
                                                    float("-inf"),          \
                                                    device = scores.device).triu(1)
            # Padding Mask（填充掩码）：对 padding 位置乘上 -1e9 从而将其 mask 掉，防止它们参与注意力计算
                # attention_mask 原来的形状 [batch_size, seq_len_kv]
                # attention_mask 对齐广播到 scores 形状：[batch_size, num_heads, seq_len, seq_len_kv]
            if attention_mask is not None:
                # attention_mask 长度对齐保护
                if attention_mask.size(-1) < scores.size(-1):
                    pad_len = scores.size(-1) - attention_mask.size(-1)
                    attention_mask = nn.functional.pad(attention_mask, (0, pad_len), value=1)
                elif attention_mask.size(-1) > scores.size(-1):
                    attention_mask = attention_mask[..., -scores.size(-1):]
                scores += (1 - attention_mask.unsqueeze(1).unsqueeze(2)) * -1e9
            
            # softmax 归一化 → 加 dropout → 对 value 加权求和
                # output 形状：[batch_size, num_heads, seq_len, head_dim]
            output = self.attn_dropout(F.softmax(scores.float(), dim = -1).type_as(xq)) @ xv

        # 把多头注意力的结果拼回成正常的特征维度
            # output 形状：[batch_size, seq_len, hidden_size]
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)

        # 在残差相加之前，对注意力分支的输出先做 dropout
        output = self.resid_dropout(self.o_proj(output))

        return output, past_kv
        
                                                    
# SwiGLU FFN
class FeedForward(nn.Module):
    def __init__(self, config: MiniMindConfig, intermediate_size: int = None):
        super().__init__()
        intermediate_size = intermediate_size or config.intermediate_size
        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias = False)     # 门控
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias = False)       # 升维
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias = False)     # 降维

        # 激活函数
        self.act_fn = ACT2FN[config.hidden_act]

        # FFN 输出层 dropout
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(
            self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
            )


# MOE FFN
class MoEFeedForward(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias = False)
        # 设置专家网络列表
        self.experts = nn.ModuleList([FeedForward(config, intermediate_size = config.intermediate_size) for _ in range(config.num_experts)])
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        batch_size, seq_len, hidden_dim = x.shape
        # 将输入矩阵展平为 x_flat 形状： [batch_size * seq_len, hidden_dim]
        x_flat = x.view(-1, hidden_dim)
        # 1. 获取每个 token 所选专家的权重、索引
            # scores 形状：[batch_size * seq_len, num_experts]
            # topk_weight 形状：[batch_size * seq_len, num_experts_per_tok]
            # topk_idx 形状：[batch_size * seq_len, num_experts_per_tok]
        scores = F.softmax(self.gate(x_flat), dim = -1)
        topk_weight, topk_idx = torch.topk(scores, k = self.config.num_experts_per_tok, dim = -1, sorted = False)
        # 2. 对选出的 Top-K 专家的权重之和归一化
        if self.config.norm_topk_prob: 
            topk_weight = topk_weight / (topk_weight.sum(dim = -1, keepdim = True) + 1e-20)
        # 3. 通过选中的专家网络进行FFN计算
        y = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            # mask 根据 topk_idx 中是否记录当前专家的索引，若有当前专家的索引，则记录当前专家需处理的 token 对应的下标
                # mask 形状：[batch_size * seq_len, num_experts_per_tok]
            mask = (topk_idx == i)
            if mask.any():
                # token_idx：取出当前专家处理的全部 token 对应的下标
                token_idx = mask.any(dim = -1).nonzero().flatten()
                # weight：取出当前专家处理的全部 token 对应的权重
                weight = topk_weight[mask].view(-1, 1)
                # 4. 对专家网络输出进行加权求和
                    # 专家计算 + 加权累加：
                    # x_flat[token_idx]：   只取出该专家负责的 token 特征
                    # expert(...)：         把筛选后的特征 扔进当前单个 FFN 专家 前向计算
                    # *weight：             乘上路由分配的权重（MoE 加权融合）
                    # .to(y.dtype)：        统一数据类型，防止精度报错
                    # y.index_add_(0, token_idx, 结果)：    按 token 索引，原位累加到最终输出张量 y
                y.index_add_(0, token_idx, weight * expert(x_flat[token_idx]).to(y.dtype))
            # 若训练时出现该专家未激活情况的兼容（该专家全程没有任何 token 分配，且处于训练模式 self.training=True）
            elif self.training:
                # 结果是 0，但计算图会挂上这个专家的参数
                y[0, 0] += 0 * sum(p.sum() for p in expert.parameters())
        
        # 计算MoE 路由负载均衡辅助损失 aux_loss → 防止专家闲置、部分专家过载
            # 只训练阶段 + 配置里开启了均衡损失权重才计算
        if self.training and self.config.router_aux_loss_coef > 0:
            # 获取每个专家被选中的平均频次（负载）：
                # one_hot：     把每个 token 选中的专家 id，转独热编码形状：
                                            # [batch_size * seq_len, num_experts_per_tok, num_experts]
                # .mean(0)：    按全体 token 维度求平均
                # load 形状：   [num_experts] →  含义：每个专家被选中的平均频次（负载）
            load = F.one_hot(topk_idx, self.config.num_experts).float().mean(0)
            # 计算均衡辅助损失：
                # scores.mean(0)：所有 token 对每个专家的平均路由打分
                # load * scores.mean(0)：负载分布 × 路由分数分布
                # .sum()：求和
                # 乘专家数、平衡系数：
                    # 约束：专家被选中频率 + 路由打分尽量均匀
                    # 避免：少数专家累死、大部分专家摸鱼
            self.aux_loss = (load * scores.mean(0)).sum()   \
                            * self.config.num_experts   \
                            * self.config.router_aux_loss_coef
        else:
            self.aux_loss = scores.new_zeros(1).squeeze()

        return y.view(batch_size, seq_len, hidden_dim)


# MoE Gate
# 只在使用共享专家网络时使用该模块
class MoEGate(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config
        self.top_k = config.num_experts_per_tok
        self.num_routed_experts = config.num_routed_experts

        self.scoring_func = config.scoring_func
        self.alpha = config.aux_loss_alpha
        self.seq_aux = config.seq_aux

        self.norm_topk_prob = config.norm_topk_prob
        self.gating_dim = config.hidden_size
        # 初始化专家权重
            # 等价于一个 nn.Linear(hidden_size, num_routed_experts, bias=False)
            # 即每个专家对应一行权重向量，拿 token 的隐藏向量去和每个专家权重做点积，得到“这个 token 该不该去这个专家”的打分。
        self.weight = nn.Parameter(torch.empty(self.num_routed_experts, self.gating_dim))
        self.reset_parameters()

    # 初始化权重，Kaiming 初始化能让一开始的打分分布不要太离谱
    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a = math.sqrt(5))

    # 前向计算
    def forward(self, hidden_states):
        batch_size, seq_len, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        # 1. 获取每个 token 所选专家的权重、索引
            # logits、scores 形状： [batch_size * seq_len, num_routed_experts]
        logits = F.linear(hidden_states, self.weight, None)
        if self.scoring_func == "softmax":
            scores = logits.softmax(dim = -1)
        else:
            raise NotImplementedError(f"insupportable scoring function for MoE gating: {self.scoring_func}")
        # topk_weight 形状：[batch_size * seq_len, num_experts_per_tok]
        # topk_idx 形状：[batch_size * seq_len, num_experts_per_tok]
        topk_weight, topk_idx = torch.topk(scores, k = self.top_k, dim = -1, sorted = False)

        # 2. 对选出的 Top-K 专家的权重之和归一化
        if self.top_k > 1 and self.norm_topk_prob:
            scores_for_aux = scores
            aux_topk = self.top_k
            # 负载均衡辅助损失，形状：[batch_size, seq_len * num_experts_per_tok]
            topk_idx_for_aux_loss = topk_idx.view(batch_size, -1)
            # (1) 按样本计算负载均衡辅助损失，每个 batch_size 是一个样本
            if self.seq_aux:
                # scores_for_seq_aux 形状：[batch_size, seq_len, num_routed_experts]
                scores_for_seq_aux = scores_for_aux.view(batch_size, seq_len, -1)
                # ce：记录这个样本里，这个专家实际上被选中了多少次，相对“均匀分配”偏了多少。
                    # ce 形状：[batch_size, num_routed_experts]
                ce = torch.zeros(batch_size, self.num_routed_experts, device = hidden_states.device)
                ce.scatter_add_(1, 
                                topk_idx_for_aux_loss, 
                                torch.ones(batch_size, seq_len * aux_topk, device = hidden_states.device), 
                                ).div_(seq_len * aux_topk / self.num_routed_experts)
                aux_loss = (ce * scores_for_seq_aux.mean(dim = -1)).sum(dim = 1).mean() * self.alpha
            # (2) 按整个 batch 全局计算负载均衡辅助损失
            else:
                # mask_ce 形状：[batch_size * seq_len * num_experts_per_tok, num_routed_experts]
                # ce 各专家被选择的(次数/总选择次数)，形状：[num_routed_experts]
                # Pi 各专家的路由打分，形状：[num_routed_experts]
                mask_ce = F.one_hot(topk_idx_for_aux_loss.view(-1), num_classes = self.num_routed_experts)
                ce = mask_ce.float().mean(0)
                Pi = scores_for_aux.mean(0)
                fi = ce * self.num_routed_experts
                aux_loss = (Pi * fi).sum() * self.alpha

        else:
            aux_loss = scores.new_zeros(1).squeeze() # 0

        return topk_idx, topk_weight, aux_loss

        
        

# MoE FeedForward with shared experts
# 只在使用共享专家网络时使用该模块
class MoEFeedForward_shared_experts(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        # 专家层
        # (1)路由专家
        self.experts = nn.ModuleList([FeedForward(config) for _ in range(config.num_routed_experts)])
        # (2)共享专家
        if config.num_shared_experts > 0:
            self.shared_experts = nn.ModuleList([FeedForward(config) for _ in range(config.num_shared_experts)])

        # 门控层
        self.gate = MoEGate(config)

    def forward(self, x):
        identity = x
        orig_shape = x.shape
        batch_size, seq_len, hidden_dim = orig_shape

        # 使用门控网络选择路由专家
        topk_idx, topk_weight, aux_loss = self.gate(x)
        # 将 x 形状转换为 [batch_size * seq_len, hidden_dim] 
        # flat_topk_idx 形状：[batch_size * seq_len * num_experts_per_tok]
        x = x.view(-1, x.shape[-1])
        flat_topk_idx = topk_idx.view(-1)

        # 训练阶段：
        if self.training:
            # 把每个 token 复制 num_experts_per_tok 份
                # x 形状：[batch_size * seq_len * num_experts_per_tok, hidden_dim]
            x = x.repeat_interleave(self.config.num_experts_per_tok, dim = 0)
            y = torch.empty_like(x, dtype = x.dtype)

            for i, expert in enumerate(self.experts):
                expert_out = expert(x[flat_topk_idx == i])
                if expert_out.shape[0] > 0:
                    y[flat_topk_idx == i] = expert_out.to(y.dtype)
                else:
                    y[flat_topk_idx == i] = expert_out.to(y.dtype)  \
                                            + 0 * sum(p.sum() for p in expert.parameters())
            # 加权求和
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim = 1)
            y = y.view(*orig_shape)

        # 推理阶段：
        else:
            y = self.moe_infer(x, flat_topk_idx, topk_weight.view(-1, 1)).view(*orig_shape)
        
        # 共享专家输出
        if self.config.num_shared_experts > 0:
            for expert in self.shared_experts:
                y = y + expert(identity)

        self.aux_loss = aux_loss


    # MoE 推理方法：
    @torch.no_grad()
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        expert_cache = torch.zeros_like(x)
        # 按专家顺序对 token 分拣，得到的 idx 为按专家排序的展开的 token 索引
        idxs = flat_expert_indices.argsort()
        # 算出每个专家负责的数据段在排序后数组里的起止下标，用来切片批量推理
            # 按专家打包，统计每个专家分到多少 token，划分批次，算出每个专家负责的 token 区间：[0~a), [a~b), [b~c)...]
            # flat_expert_indices.bincount()：  统计每个专家被分配到多少条路由记录，得到一个数组：[专家0条数, 专家1条数, 专家2条数, ...]
            # .cpu().numpy()：                  转到 CPU 并转成 numpy 数组，方便做累加
            # .cumsum(0)                        沿第 0 维前缀和累加，得到每段结束的索引位置，比如 [2,2,1] → 累加后 [2,4,5]，表示：
                # 专家 0：0~2
                # 专家 1：2~4
                # 专家 2：4~5
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum()
        # 记录计算排序后的每一条路由，到底属于原始哪个 token 索引
            # 因为每个 token 分配到 num_experts_per_tok 个专家，idx 为展开后的索引，需要除以 num_experts_per_tok 得到原始 token 索引
        token_idxs = idxs // self.config.num_experts_per_tok

        for i, end_idx in enumerate(tokens_per_expert):
            # 计算当前包的起始位置
            start_idx = 0 if i == 0 else tokens_per_expert[i - 1]
            if start_idx == end_idx:
                continue

            # 取当前包对应专家
            expert = self.experts[i]
            # 取出 token 对应的原始索引
            epx_token_idx = token_idxs[start_idx: end_idx]
            # 取出对应 token 原始数据
            expert_tokens = x[epx_token_idx]
            # 专家一次性处理所有对应 token
            expert_out = expert(expert_tokens).to(expert_cache.dtype)
            # 加权求和
            expert_out.mul_(flat_expert_weights[idxs[start_idx: end_idx]])

            # 将该专家输出写入缓存
            expert_cache.scatter_add_(0, 
                                      epx_token_idx.view(-1, 1).repeat(1, x.shape[-1]), 
                                      expert_out)
            
        return expert_cache


# Transformer Block
class MiniMindBlock(nn.Module):
    def __init__(self, layer_id: int, config: MiniMindConfig):
        super().__init__()
        self.self_attn = Attention(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps = config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps = config.rms_norm_eps)
        self.mlp = FeedForward(config)  # if not config.use_moe else MoEFeedForward(config)

    def forward(self, 
                hidden_states, 
                position_embeddings, 
                past_key_value = None, 
                use_cache = False, 
                attention_mask = None):
        residual = hidden_states

        # Self-Attention
        hidden_states, present_key_value = self.self_attn(self.input_layernorm(hidden_states), 
                                                        position_embeddings,
                                                        past_key_value,
                                                        use_cache, 
                                                        attention_mask)
        hidden_states += residual

        # FeedForward
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))

        return hidden_states, present_key_value


# MiniMind Model
class MiniMindModel(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.num_hidden_layers = config.num_hidden_layers
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([MiniMindBlock(layer_id, config) for layer_id in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps = config.rms_norm_eps)
        freqs_cos, freqs_sin = precompute_freqs_cis(dim = config.head_dim, 
                                                    end = config.max_position_embeddings, 
                                                    rope_base = config.rope_theta, 
                                                    rope_scaling = config.rope_scaling)
        self.register_buffer("freqs_cos", freqs_cos, persistent = False)
        self.register_buffer("freqs_sin", freqs_sin, persistent = False)

    def forward(self, 
                input_ids, 
                attention_mask = None, 
                past_key_values = None,
                use_cache = False, 
                **kwargs):
        batch_size, seq_lenth = input_ids.shape
        # 兼容处理：若传入缓存格式并非 layers 格式，则重置缓存
        if hasattr(past_key_values, 'layers'):
            past_key_values = None
        # 若传入缓存为空，缓存空间初始化为全 None
        past_key_values = past_key_values or [None] * len(self.layers)
        # 计算当前 token 的全局位置起点位置，即取第一层第一个张量的第 2 维长度，也就是已经生成的 token 数量
        start_pos = (past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0)
        # 获取input_ids[batch_size, seq_lenth]的嵌入表示，形状为[batch_size, seq_lenth, hidden_size]，再做 dropout 处理
        hidden_states = self.dropout(self.embed_tokens(input_ids))
        # 从历史长度接着切到当前段长度，把 RoPE 需要的 cos/sin 切片计算出来，并且按当前绝对位置对齐
        position_embeddings = (self.freqs_cos[start_pos: start_pos + seq_lenth], 
                               self.freqs_sin[start_pos: start_pos + seq_lenth])
        
        # 大模型逐层做前向传播 + KV 缓存推理：
            # 准备一个列表，用来收集每一层返回的新缓存
        presents = []
            # 逐层前向，把第 i 层 Transformer 和第 i 层对应的历史 KV 缓存一一配对(layer_i, past_kv_i)
        for layer, past_key_value in zip(self.layers, past_key_values):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value = past_key_value,
                use_cache = use_cache,
                attention_mask = attention_mask
            )
            # 把这一层算出来的缓存保存到 presents 列表里。 append()：列表（list）的方法：在列表的最后面，添加一个元素
            presents.append(present)
        hidden_states = self.norm(hidden_states)
        # aux_loss = sum([layer_id.mlp.aux_loss for layer_id in self.layers if isinstance(layer_id.mlp, MOEFeedForward)], hidden_states.new_zeros(1).squeeze())
        aux_loss = hidden_states.new_zeros(())
        
        return hidden_states, presents, aux_loss



class MiniMindForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = MiniMindConfig
    def __init__(self, config: MiniMindConfig = None):
        self.config = config or MiniMindConfig()
        super().__init__(config)
        self.model = MiniMindModel(self.config)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias = False)
        # 词嵌入层（embed_tokens）和语言模型输出层（lm_head）的权重绑定共享，不额外开辟参数
            # 大幅减少参数量：词表通常很大（几万～几十万），绑定能省掉一层全连接参数
            # 保持输入输出语义空间一致：让 “输入编码” 和 “输出解码” 在同一个向量空间。
            # 训练更稳定，收敛更快。
        self.model.embed_tokens.weight = self.lm_head.weight

    def forward(self, 
                input_ids,
                attention_mask = None,
                past_key_values = None,
                use_cache = False,
                logits_to_keep = 0,
                labels = None,
                **kwargs):
        # hidden_states, past_key_values, aux_loss
        hidden_states, past_key_values, aux_loss = self.model(input_ids, 
                   attention_mask = attention_mask, 
                   past_key_values = past_key_values, 
                   use_cache = use_cache, 
                   **kwargs)
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        # 每个位置预测每个词的概率分数，logits.shape = [batch_size, seq_len, vocab_size]，vocab_size中储存了分数
        logits = self.lm_head(hidden_states)[:, slice_indices, :]
        loss = None

        # 只有在传入标签时，才会计算训练损失。labels 就是“打开训练模式”的开关。
        if labels is not None:
            # 计算下一个 token 预测的交叉熵损失
            # x 去掉最后一个位置的输出, y 去掉第一个位置的标签, .contiguous()让张量在内存里连续排布
            x, y = logits[..., :-1, : ].contiguous(), labels[..., 1:].contiguous()
            loss = nn.functional.cross_entropy(x.view(-1, x.size(-1)), y.view(-1), ignore_index = -100)
        
        # MoeCausalLMOutputWithPast(loss=loss, aux_loss=aux_loss, logits=logits, past_key_values=past_key_values, hidden_states=hidden_states)
        output = CausalLMOutputWithPast(loss = loss,
                                        logits = logits,
                                        past_key_values = past_key_values,
                                        hidden_states = hidden_states)
        output.aux_loss = aux_loss

        return output

