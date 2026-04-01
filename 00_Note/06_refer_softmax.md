# **为什么 Transformer 自注意力里 softmax 必须除以 $\sqrt{d_k}$，否则模型会训练不起来**

---

## 1. 先搞懂核心概念：softmax 是什么？
softmax 是一个把**任意数值**转换成「概率分布」的函数，公式是：
$$
\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_{j=1}^n e^{x_j}}
$$
它的特点是：
- 所有输出加起来等于 1，像概率一样
- 输入的数值差越大，输出越极端：
  - 大的输入会被放大到**接近 1**
  - 小的输入会被压缩到**接近 0**

---

## 2. 关键问题：softmax 的梯度为什么会消失？
### 梯度是什么？
梯度是神经网络训练时，用来**更新参数的“方向信号”**。梯度越接近 0，信号越弱，模型就学不动了，也就是「梯度消失」。

### softmax 梯度的特性
softmax 的梯度公式（对输入 $x_i$ 求导）：
$$
\frac{\partial \text{softmax}(x_i)}{\partial x_j} =
\begin{cases}
\text{softmax}(x_i)(1-\text{softmax}(x_i)), & i=j \\
-\text{softmax}(x_i)\text{softmax}(x_j), & i \neq j
\end{cases}
$$

我们看这个公式：
- 当输出 $\text{softmax}(x_i) \approx 1$ 时：$1 \times (1-1) = 0$，梯度直接变成 0
- 当输出 $\text{softmax}(x_i) \approx 0$ 时：$0 \times (1-0) = 0$，梯度也直接变成 0

**结论**：softmax 一旦输出极端（0 或 1），梯度就没了，模型参数再也更新不了，训练直接“躺平”。

---

## 3. 为什么不除以 $\sqrt{d_k}$ 会导致输出极端？
我们之前算过：
- Query 和 Key 做点积 $QK^\top$ 时，结果的**方差会变成 $d_k$**（维度越大，数值波动越大）
- 当 $d_k$ 很大（比如 64、128），点积结果会出现**极大的数**和**极小的数**

把这些极端数值输入 softmax：
- 极大的数 → softmax 输出接近 1
- 极小的数 → softmax 输出接近 0
- 最终梯度全部变成 0 → 模型无法有效训练

---

## 4. 除以 $\sqrt{d_k}$ 是怎么解决问题的？
除以 $\sqrt{d_k}$ 后：
- 点积的方差被**归一化回 1**，数值回到正常范围
- softmax 输入不再极端，输出分布更平滑（不会出现 0/1）
- 梯度保持在有效范围，模型能正常更新参数，顺利训练

---

## 💡 一句话总结
**不缩放 → 点积数值爆炸 → softmax 输出极端（0/1）→ 梯度消失 → 模型学不动；
除以 $\sqrt{d_k}$ → 数值归一化 → softmax 输出平滑 → 梯度正常 → 模型能训练。**

---



# softmax 梯度公式的完整推导与计算

## 一、先明确 softmax 的定义
softmax 函数的作用是把一个**实数向量** $x = [x_1, x_2, ..., x_n]^T$ 映射成一个**概率分布**（所有输出和为1，且每个输出在 $(0,1)$ 之间）。
对于第 $i$ 个输出，定义为：
$$
\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_{k=1}^n e^{x_k}}
$$
我们记 $s_i = \text{softmax}(x_i)$，简化符号，即 $s_i = \frac{e^{x_i}}{S}$，其中 $S = \sum_{k=1}^n e^{x_k}$ 是所有指数项的和。

---
## 二、分情况求导：推导梯度公式
我们要求的是 $\frac{\partial s_i}{\partial x_j}$，也就是**第 $i$ 个 softmax 输出，对第 $j$ 个输入的偏导数**。
根据求导的商法则 $\frac{\partial}{\partial x}\left(\frac{u}{v}\right) = \frac{u'v - uv'}{v^2}$，我们分两种情况讨论：

### 情况1：$i = j$（对自身输入求导）
此时 $u = e^{x_i}$，$v = S = \sum_{k=1}^n e^{x_k}$
- 分子部分：$u'v - uv' = e^{x_i} \cdot S - e^{x_i} \cdot \frac{\partial S}{\partial x_i}$
- 因为 $\frac{\partial S}{\partial x_i} = e^{x_i}$（只有 $k=i$ 这一项对 $x_i$ 求导不为0）
- 代入得：$e^{x_i}S - e^{x_i} \cdot e^{x_i} = e^{x_i}(S - e^{x_i})$
- 分母部分：$v^2 = S^2$

因此：
$$
\frac{\partial s_i}{\partial x_i} = \frac{e^{x_i}(S - e^{x_i})}{S^2} = \frac{e^{x_i}}{S} \cdot \frac{S - e^{x_i}}{S} = s_i \cdot \left(1 - \frac{e^{x_i}}{S}\right) = s_i(1 - s_i)
$$
这就对应了公式里 $i=j$ 的情况：$\text{softmax}(x_i)(1-\text{softmax}(x_i))$

---
### 情况2：$i \neq j$（对其他输入求导）
此时 $u = e^{x_i}$，$v = S = \sum_{k=1}^n e^{x_k}$
- 分子部分：$u'v - uv' = 0 \cdot S - e^{x_i} \cdot \frac{\partial S}{\partial x_j}$（因为 $i \neq j$，$e^{x_i}$ 对 $x_j$ 求导为0）
- 而 $\frac{\partial S}{\partial x_j} = e^{x_j}$（只有 $k=j$ 这一项对 $x_j$ 求导不为0）
- 代入得：$- e^{x_i} \cdot e^{x_j}$
- 分母部分：$v^2 = S^2$

因此：
$$
\frac{\partial s_i}{\partial x_j} = \frac{- e^{x_i} e^{x_j}}{S^2} = - \frac{e^{x_i}}{S} \cdot \frac{e^{x_j}}{S} = - s_i s_j
$$
这就对应了公式里 $i \neq j$ 的情况：$-\text{softmax}(x_i)\text{softmax}(x_j)$

---
## 三、公式的矩阵形式（更直观）
我们可以把所有偏导数写成一个**雅可比矩阵** $J \in \mathbb{R}^{n \times n}$，其中 $J_{ij} = \frac{\partial s_i}{\partial x_j}$：
$$
J = 
\begin{bmatrix}
s_1(1-s_1) & -s_1s_2 & \dots & -s_1s_n \\
-s_2s_1 & s_2(1-s_2) & \dots & -s_2s_n \\
\vdots & \vdots & \ddots & \vdots \\
-s_ns_1 & -s_ns_2 & \dots & s_n(1-s_n)
\end{bmatrix}
$$
可以简化为矩阵形式：$J = \text{diag}(s) - s s^T$，其中 $\text{diag}(s)$ 是把向量 $s$ 放在对角线上的对角矩阵，$s s^T$ 是向量的外积。

---
## 四、图中提到的梯度特性：为什么会“梯度消失”？
我们再看你图里的两个结论，用公式验证一下：
1.  **当 $s_i \approx 1$ 时**：
    代入 $i=j$ 的公式：$s_i(1-s_i) \approx 1 \times (1-1) = 0$
    同时，对于所有 $j \neq i$，$-s_i s_j \approx -1 \times 0 = 0$（因为 $\sum s_k=1$，$s_i\approx1$ 时其他 $s_j\approx0$）
    → 整个第 $i$ 行的梯度几乎全为0，模型无法再更新这个神经元的参数。

2.  **当 $s_i \approx 0$ 时**：
    代入 $i=j$ 的公式：$s_i(1-s_i) \approx 0 \times (1-0) = 0$
    同时，对于所有 $j \neq i$，$-s_i s_j \approx -0 \times s_j = 0$
    → 整个第 $i$ 行的梯度也几乎全为0，模型同样无法更新这个神经元的参数。

这就是 softmax 函数的一个重要特性：**当输出概率极端接近0或1时，梯度会趋近于0，导致梯度消失，模型难以训练**。

---
## 五、举个数值例子，帮你直观计算
假设输入向量 $x = [1, 2, 3]$，我们手动计算梯度：
1.  先算 softmax 输出：
    $e^{x} = [e^1, e^2, e^3] \approx [2.718, 7.389, 20.086]$
    $S = 2.718 + 7.389 + 20.086 = 30.193$
    $s = [\frac{2.718}{30.193}, \frac{7.389}{30.193}, \frac{20.086}{30.193}] \approx [0.090, 0.245, 0.665]$

2.  计算雅可比矩阵：
    - 对角线（$i=j$）：
      $J_{11} = 0.090 \times (1-0.090) \approx 0.082$
      $J_{22} = 0.245 \times (1-0.245) \approx 0.185$
      $J_{33} = 0.665 \times (1-0.665) \approx 0.223$
    - 非对角线（$i \neq j$）：
      $J_{12} = J_{21} = -0.090 \times 0.245 \approx -0.022$
      $J_{13} = J_{31} = -0.090 \times 0.665 \approx -0.060$
      $J_{23} = J_{32} = -0.245 \times 0.665 \approx -0.163$

    最终矩阵：
    $$
    J \approx 
    \begin{bmatrix}
    0.082 & -0.022 & -0.060 \\
    -0.022 & 0.185 & -0.163 \\
    -0.060 & -0.163 & 0.223
    \end{bmatrix}
    $$
    可以验证：每一行的和都为0（因为 $\sum_j \frac{\partial s_i}{\partial x_j} = 0$，softmax 输出和恒为1，对所有输入求导的和为0）。

---
## 六、补充：为什么这个梯度特性很重要？
在深度学习中，softmax 通常和交叉熵损失一起使用（分类任务的标准组合），**交叉熵损失会抵消 softmax 的梯度消失问题**：
交叉熵损失 $L = -\sum_i y_i \log s_i$，对输入 $x_j$ 求导后，结果为 $\frac{\partial L}{\partial x_j} = s_j - y_j$，形式非常简洁，且不会出现梯度消失的问题。
这也是为什么分类任务中，softmax + 交叉熵是黄金组合的核心原因之一。

---
## 💡 总结
1.  softmax 梯度分两种情况：自身输入求导为 $s_i(1-s_i)$，其他输入求导为 $-s_i s_j$
2.  当输出概率极端接近0或1时，梯度会趋近于0，导致梯度消失
3.  实际训练中，交叉熵损失会抵消这个问题，让梯度保持有效

