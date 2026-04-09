# FFN（Feed Forward Network）前馈神经网络

## FFN 的作用
目的：把 attention 已经聚合来的信息，进一步变成“可用的、可表达的、适合下一层处理”的特征。  
- 提升表达能力：
  Attention 更擅长“把别处的信息拿过来”，但拿过来之后，还需要在当前 token 内部做复杂变换。FFN 通过升维、非线性、再降维，让模型能表示更复杂的函数关系
- 做特征重组与筛选：
  在每个 token 的向量内部，决定哪些特征该放大、哪些该抑制、哪些该组合成新特征。MiniMind 里用的 SwiGLU，本质上就是这种“门控式筛选”。
- 把上下文信息加工成“下一层能继续用”的形式
  Attention 得到的是“上下文混合后的表示”，FFN 会把这些表示重新编码，让下一层更容易继续抽象、推理、生成。

隐藏向量是 token 的“语义坐标”，FFN 作为“特征生成器”升维以后，把这个 token 投影到一个更大的特征空间里，让模型有机会激活更多“特征基”：
- 这个 token 具备哪些局部属性
- 哪些属性要保留
- 哪些属性要抑制
- 哪些属性可以组合成更高级的语义


FFN 让训练更容易:
- 如果只有 attention，模型主要在做 token 之间的加权混合，表达会偏“线性组合”；FFN 让网络可以学到更复杂的非线性边界
- 在每层做一点局部改写、抽象、筛选，逐层叠加，不必指望一层就完成复杂推理


整个生成链条：
- Attention：把相关上下文搬过来
- FFN：把搬来的信息整理成可供决策的内部状态
- 输出层：根据这个状态决定下一个 token 的概率分布


## 对比标准 FFN 和增强 FFN 
|    特性    |    标准 FFN    |    增强 FFN（Gated）    |
|:----------:|:-------------:|    :-------------:     |
|  激活函数   |  ReLU/GELU    | GELU / SiLU / Swish 等 |
|  中间结构   |   单路计算    |  双路计算(乘积)          |
|   参数量   |      较少      |  多一个 linar 层        |
| 非线性能力 |      一般      |   更强(引入乘法门控)    |
|  性能表现  |      一般      |   更强( SOTA 标配)    |

## 激活函数完整对比表

| 激活函数 | 公式 | 输出范围 | 特点 | 应用场景 |
|:----------:|:------:|:---------:|:------:|:---------:|
| Sigmoid | $\displaystyle \sigma(x)=\frac{1}{1+e^{-x}}$ | $(0,1)$ | 平滑、输出可看作概率；易梯度消失、非零均值 | 二分类输出层、早期RNN门控 |
| ReLU | $\max(0,x)$ | $[0,+\infty)$ | 计算极快；负半轴硬截断，易出现死神经元 | CNN、传统深度网络隐藏层 |
| Leaky ReLU | $\begin{cases}x,&x>0\\ 0.01x,&x\le 0\end{cases}$ | $(-\infty,+\infty)$ | 保留微弱负梯度，避免神经元死亡 | 替代ReLU，用于深层网络 |
| ELU | $\begin{cases}x,&x>0\\ \alpha(e^x-1),&x\le 0\end{cases}$ | $(-\alpha,+\infty)$ | 平滑、负值区有界，均值接近0 | 深层网络、希望更稳定训练 |
| SiLU / Swish | $x\cdot\sigma(x)$ | $(-\infty,+\infty)$ | 平滑自门控，负值区非零，梯度柔和 | LLM、Transformer、SwiGLU门控 |
| GELU | $x\cdot\Phi(x),\ \Phi=\text{正态CDF}$ | $(-\infty,+\infty)$ | 概率式软门限，平滑稳定 | BERT、GPT、标准Transformer FFN |
| **SwiGLU** | $\sigma(xW_1)\otimes(xW_2)\cdot W_3$ | $(-\infty,+\infty)$ | 门控线性单元，表达能力强，参数量更大 | LLaMA、现代大模型FFN（增强版） |



## minimind 中的 FFN (SwiGLU)

minimind 中使用的 FFN 是增强 FFN（Gated），即 SwiGLU 结构：
- 线性层：
  - gate_proj
  - up_proj
  - down_proj
- 信息支路
  - 门控分支
  - 信息分支


SwiGLU FFN 的标准形式：
$$ FFN(x) = W_2(SiLU(W_1x) \odot (W_3x)) $$

$$ SiLU(x) = x * sigmoid(x) = x * \sigma(x) = x * \frac{1}{1 + e^{-x}} $$

其中：
- 信息支路
  - 门控分支升维: $x → W_1x$
  - 信息分支升维: $x → W_3x$
- $SiLU(W_1x)$ = 门
- $W_3x$ = 信息
- $\odot$ = 控制信息流
- $W_3x$ = 降维



## 维数拓展
维数拓展代码段：
``` python
self.intermediate_size = kwargs.get("intermediate_size" math.ceil(hidden_size * math.pi / 64) * 64) # 本行在class MiniMindConfig(PretrainedConfig)中 
 intermediate_size = intermediate_size or config.intermediate_size
```

历史版本代码段：
``` python
if config.intermediate_size is None: 
  intermediate_size = int(config.hidden_size * 8 / 3)
  config.intermediate_size = 64 * ((intermediate_size + 64 - 1) // 64)
```


### $\pi$ 与 $\frac{8}{3}$
当前版本使用 $\pi ≈ 3.1416$ 来计算中间层维度，历史版本使用 $\frac{8}{3} ≈ 2.6667$。  
对 `hidden_size = 768` 来说，`768 × 8/3 = 2048`，而 `768 × π ≈ 2412.74`，再向上补到 `64` 的倍数后变成 `2432`，比历史版本大 `384`，也就是大约 `18.75%`。
- 历史版本里的 `8/3` 不是随便拍的，它和 [SwiGLU/GLU 论文](https://arxiv.org/pdf/2002.05202)里的做法一致：`GLU` 变体比原始 `FFN` 多了一个权重矩阵，所以把隐藏层宽度缩小到原来的 `2/3`，这样可以让参数量和计算量大致保持不变；论文里甚至直接给出 `d_model=768` 时，原始 `FFN` 是 `3072`，`GLU` 变体缩到 `2048`。

- MiniMind 代码直接把倍率换成了 π: 
  - 更宽一点
  - 容量更大一点
  - 但代价是参数和计算也更高一点

### `math.ceil(...)` 和 `//` 
- `math.ceil(...)`: 向上取整
- `//` 向下取整  

在当前版本和历史版本中取整升维操作方式的解读：
- 当前版本：
  为获取略大于或等于 $hidden_size * \pi$ 能被 `64` 整除的数 `intermediate_size`，这个数除以 `64` 后的比值必然是比 $hidden_size * \pi$ 除以  `64` 后大于或等于的第一个整数。
  $$ 0 \leq \frac{hidden_size * \pi}{64} - \frac{intermediate_size}{64} \lt 1 $$

- 历史版本：
  先获取 $hidden_size * 8/3$ 的整数部分，为得到略大于或等于该数且能被 `64` 整除的数 `intermediate_size`，采取加 1 后向下取整以获取`intermediate_size`与`64`的最近整数倍数。