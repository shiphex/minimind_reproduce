# GQA框架
# `GQA`原理

## `GQA` 与 `MHA`
- 《Attention Is All You Need》里的 `MHA`（Multi-Head Attention 多头注意力），核心思想是：  
  - 把同一个输入投影成多组 $Q,K,V$，每一组头各自做注意力，再把结果拼起来。
- `GQA`（Grouped-`Query` Attention 分组查询注意力）： 
  - `Query` 头很多
  - `Key` / `Value` 头更少
  - 多个 `Query` 头 共享 同一组 K/V

如果把 attention 看成一个“查阅上下文并融合信息”的系统：
- `MHA`：每个头都完整配套地查
- `GQA`：保留多头查询能力，但让多个头共享查询到的“资料库”（K/V）

`GQA` 相比于 `MHA` 的优缺点：
- 优点：
  - 显存占用大幅下降，减小 KV Cache（历史 K/V 的存储）
  - 推理速度更快，$QK^\top$ 计算时 $K$ 因为分组共享从而头数更少，矩阵乘法规模更小
    - 减少显存访问
    - 减少计算
    - 更高吞吐  
  - 更容易扩展上下文长度，KV减少可拓展更多上下文
- 缺点：
  - 表达能力下降，`MHA` 中每个 head 有独立的 K/V 可学习完全不同的注意模式，`GQA` 数个 `Query` 头共享同一组 K/V，只能学习到一种注意模式
  - 注意力多样性减少，不同 head 学不同模式（语法 / 指代 / 远程依赖等），多个 head 用同一组 K/V ，attention 的“多视角能力”被压缩了
  - 训练时可能更难调优，收敛慢


## 在整个系统中发挥什么作用？
让每个 token 不再只看自己，而是能从上下文中取信息：
- 建立 token 和 token 之间的联系
- 找出“当前这个位置最该关注谁”
- 把相关位置的信息加权汇总回来
- 让模型具备长距离依赖建模能力


## 输入的数据是什么？
对于一层 attention 来说，输入通常是上一层输出的 token 表示向量序列矩阵：
$$ X \in \mathbb{R}^{B \times T \times d_{model}} $$
- $X$ 是每个 token 的“当前理解状态”
- $B$ 是 batch size 批量大小
- $T$ 是序列长度、seq_len
- $d_{model}$ 是每个 token 的隐藏维度，也叫模型维度，比如 512、768、1024 等等  
``` python
X : (B, T, 512)
```



## 数据在其中进行了怎样的运算？
### 1、线性变化生成 $K$、$Q$、$V$  
$X$ 通过线性变换`映射`生成三类东西：
$$  \begin{array}{rcl} Q = XW_Q,  &   K = XW_K,  &   V = XW_V \end{array} $$
- $Q$：`Query`，查询
- $K$：`Key`，键
- $V$：`Value`，值
- $Q$、$K$、$V$ 都是可训练参数
- 这些投影把同一个输入 $X$ 映射成不同“角色”的表示  

其中输入 $X$ 是一个 $B \times T \times d_{model}$ 的矩阵，每个 token 的隐藏表示。  
$$ X \in (B, T, d_{model}) = (B, T, 512) $$
假设：`映射`矩阵 $W_Q$、$W_K$、$W_V$ 都是 $head\_dim \times head\_dim$ 的矩阵
- 那么 head 数就会是 $d_{model} // head\_dim = 512 // 64 = 8$ 个头
- 那么生成的每个头 $Q_i$、$K_i$、$V_i$ 都是 $B \times T \times head\_dim$ 的矩阵。  
  $$ Q_i, K_i, V_i : (B, T, 64) $$    
然后它会做：
  $$ Attention_i = softmax(Q_i K_i^\top)V_i $$


### 2、把头拆开
**核心问题：** `GQA` 中 $Q$ 的 head 数目多，$K$、$V$ 的 head 数目少，$K$、$V$ 分组共享 $Q$ 的 head  
解决方式：通过 reshape + repeat（或 broadcast）让 Q 去“对齐”较少的 K/V 头  

1. 标准 `MHA` 的Q/K/V对应，假设：
  - hidden_dim = d_model = 512
  - num_heads = 8
  - head_dim = d_model // num_heads = 64  
  则：
  $$ Q, K, V : (B, T, 8, 64) $$  
  每个 head 都有完整的Q/K/V对应

2. `GQA` 的Q/K/V对应，假设：
  - hidden_dim = d_model = 512 
  - Q_heads = 8
  - KV_heads = 2
  - head_dim = d_model // Q_heads = 64    
  在线性投影后：
  ``` python
  Q : (B, T, 8, 64)
  K : (B, T, 2, 64)   # Q 与 K/V 不能一一对应
  V : (B, T, 2, 64)   
  ```
  Q 与 K/V 不能一一对应


**解决办法：** 把 K/V “复制（repeat）”到和 Q 一样多的头数：  
``` python
# 计算重复因子
repeat_factor = Q_heads // KV_heads     # 8 // 2 = 4

K = K.repeat_interleave(4, dim=2)   # K → repeat 成 (B, T, 8, 64)
V = V.repeat_interleave(4, dim=2)   # V → repeat 成 (B, T, 8, 64)
```
伪装成 `MHA` ，变成：
``` text
K: [k1, k1, k1, k1, k2, k2, k2, k2]
V: [v1, v1, v1, v1, v2, v2, v2, v2]
```
> 注：一般形状：[batch, channels, height, width]
> 即：Q/K/V：[batch, seq_len, num_heads, head_dim]

K/V 会被缓存下来（KV cache）,每一步只算 Q，直接减少 4 倍显存：
``` text
MHA：
    KV cache: (T, 8, 64)

GQA：
    KV cache: (T, 2, 64)
```
> **注意：**GQA 并没有改变 attention 的数学形式，它只是通过“减少 KV 头 + 复制对齐”来实现一种参数共享结构。

> 为什么使用多头注意力？
- 多个 head 往往更强、更稳、更容易学到不同关系
- 多头 attention 允许模型同时在多个不同的相似性空间里做匹配  

Attention 做的事情是：
$$ Attention = softmax(Q K^\top)V $$
重点是$Q K^\top$，表示当前 token 需要关注哪些位置  
- 如果只有一个 head ，只能产生：
  - 一张注意力图
- 多头 attention 的核心优势是：
  - 每个 head 可以学习不同的 Q/K/V 投影，于是每个 head 可以看到不同的“关系视角”
  - 让多个较小子空间分别学习不同匹配规则：
      - head1 可以专门找主谓关系
      - head2 可以专门找指代对象
      - head3 可以专门找局部邻近信息
      - head4 可以专门找长程依赖

> 问题：为什么不能用少数个 Q head，去分别对应很多套不同的 K/V 关系？  
- Q head 少，表示“提问视角少”
- K/V head 多，表示“记忆空间多”  
如果提问视角少，你就算有很多 K/V 子空间，最后也未必都能被有效区分；这些不同的 K/V 可能会被同一个 Q 混成一团。

> 问题：为什么“复制 K/V”不会完全损失表达能力？它到底损失了哪些自由度？  
- 为什么“复制 K/V”不会完全损失表达能力？
  - 重点是$Q K^\top$
  - 即使 K/V 一样，只要 Q 不一样，不同 head 仍然会得到不同的注意力分布
  - head1 的问题方式不同
  - head2 的问题方式不同
  - 它们虽然查的是同一份“资料库”
  - 但查法不同，所以结果不同
- 它到底损失了哪些自由度？
  - 每个头不能再独立学习自己专属的 K/V 空间
  - 这套索引只能有一种
  - 两个头只能用不同的 Q 去问
  - 但不能各自重新定义“索引规则”
- 为什么这通常还能接受？
  - 因为很多头的独立性其实是冗余的
  - 保留更多 Q 的多样性，减少 K/V 的重复建模


### 3、计算注意力分数
对每个 Query 头，都会和对应的 Key 做点积，得到相关性分数：
$$ score = Q K^\top $$
通常除以一个缩放因子：
$$ \frac{Q K^\top}{\sqrt{d_k}} $$
最后做 `softmax`:
$$ A = softmax(\frac{Q K^\top}{\sqrt{d_k}}) $$
$A$ 是注意力权重矩阵，表示：当前 token 应该关注哪些位置，以及关注多少

> 为什么要除以缩放因子？  

$ d_k $是每个注意力头的维度 head_dim（总隐藏维度除以头数，比如 512/8=64）  
1. 控制方差，避免 `softmax` 梯度消失  
   假设 Query 和 Key 的每个元素都是均值为 0、方差为 1 的独立随机变量：  
  - 计算点积 $Q K^\top$ 时，维度为 $d_k$ 的向量点积，其结果方差会等于 $d_k$
  - 当 $d_k$ 很大的时候(minimind 中 head_dim 为 64，大模型中会更大)，点积结果的绝对值会非常大，方差会放大到 $d_k$
  - `softmax` 函数的特性是：输入值越大，其输出会越趋向于 one-hot 分布（某一项趋向于 1，其余趋向于 0）
  - `softmax` 的梯度在接近于 0 或 1 时，会趋向于 0 梯度消失，导致模型无法训练
  - 除以 $\sqrt{d_k}$ 可使点积的方差被归一化到 1：
  $$ Var(\frac{Q K^\top}{\sqrt{d_k}}) = \frac{Var(Q K^\top)}{d_k} = 1 $$
2. 让注意力分数的分布更 “平滑”
- 不缩放时 $d_k$ 越大，点积结果越大，`softmax` 会让模型 “过度自信”：只给极少数 token 极高权重，其余几乎忽略，丢失上下文信息。
- 缩放后，分数被拉回合理区间，`softmax` 输出的权重分布更平滑，模型能更合理地分配注意力，捕捉长距离依赖。

> 关于方差与点积：  
- 独立变量乘积的方差
$$ Var(XY) = Var(X) * Var(Y) $$
- 独立变量求和的方差
$$ Var(X+Y) = Var(X) + Var(Y) $$
- 常数乘积的方差(C 为常数)
$$ Var(C X) = C^2*Var(X) $$

点积 $Q K^\top$ 的计算公式为：
$$  \begin{array}{rcl}
Var(QK^\top) & = & Var(\sum_{i=1}^{d_k}{q_i k_i}) \\ \\
             & = & \sum_{i=1}^{d_k}Var({q_i k_i}) \\ \\
             & = & \sum_{i=1}^{d_k}Var(q_i) * Var(k_i) \\ \\
             & = & \sum_{i=1}^{d_k}(1*1) \\ \\
             & = & d_k 
\end{array}$$

> 关于 `softmax`：  
`softmax` 是一个把任意数值转换成「概率分布」的函数，公式是：  
$$ softmax(x_i) = \frac{e^{x_i}}{\sum_{j=1}^{d_k}e^{x_j}} $$
`softmax` 的特点是：
- 所有输出加起来等于 1，像概率一样
- 输入的数值差越大，输出越极端
- 大的输入会被放大到接近 1
- 小的输入会被压缩到接近 0

> 问题：若不控制方差，`softmax` 的梯度为什么会消失？   

LLM给出的关于 `softmax` 的梯度消失问题的计算：  
[_为什么 Transformer 自注意力里 softmax 必须除以 $\sqrt{d_k}$？_](06_refer_softmax.md)


### 4、用权重去加权汇总 `Value`
- 有了注意力权重后，就去乘 Value：  
$$ O = AV = softmax(\frac{Q K^\top}{\sqrt{d_k}})V $$
- 把这些位置的 V 信息加权求和  
- 得到当前位置新的表示


### 5、多头结果合并
不同头会得到不同的输出：
- 有的头关注语法关系
- 有的头关注指代关系
- 有的头关注局部邻近信息
- 有的头关注长程依赖  
最后把各头结果拼接、再线性变换：
$$ Y = Concat(O_1, O_2, ..., O_h)W_o $$
得到这一层 attention 的最终输出。

## 输出的数据是什么？
输出仍然是一个 token 序列表示矩阵：
$$ Y \in \mathbb{R}^{B \times T \times d_{model}}$$
它和输入形状通常一样，但内容已经变了：
- 原输入只是“每个 token 自己的初始表示”
- 输出则是“融入了上下文信息之后的表示”
也就是说，输出里的每个 token 向量，已经包含了它对上下文的理解。


# 其他
## Mask
