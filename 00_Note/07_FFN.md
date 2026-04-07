# FFN（Feed Forward Network）前馈神经网络

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




