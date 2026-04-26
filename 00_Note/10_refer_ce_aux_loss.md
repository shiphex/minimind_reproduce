# 序列级辅助损失计算（seq_aux = True）


## 1 先把除法式子拆开
```python
.div_(seq_len * aux_topk / self.num_routed_experts)
```
等价于：
$$\boldsymbol{ce}
= \boldsymbol{ce} \;\div\; \frac{S\cdot K}{E}
= \boldsymbol{ce} \;\times\; \frac{E}{S\cdot K}$$

符号约定：
- $S = \text{seq_len}$ 单条序列 token 数
- $K = \text{aux_topk}$ 每个 token 选多少个辅助专家
- $E = \text{num_routed_experts}$ 总路由专家数

---

## 2 先看 scatter_add 完的原始 ce 是什么
scatter_add 之后：
$$ce[b,e] = \textbf{专家}e\textbf{被选中的【绝对次数】}$$

全局总选择次数（一个样本内）：
$$\text{TotalSelect} = S \times K$$

---

## 3 关键：我们想要的是「相对负载」而不是次数
MoE 负载均衡核心假设：
> **所有专家完全均匀分配时，每个专家的期望选中次数 = $\dfrac{\text{总选择次数}}{\text{专家总数}}$**

也就是**理论均匀负载基准**：
$$\text{uniform\_count} = \frac{S\cdot K}{E}$$

---

## 4 为什么一定要除以 $\boldsymbol{E}$？

### ① 不除 E 会发生什么？
只除以 $S\cdot K$：
$$ce_{\text{raw}} = \frac{\text{专家}e\text{选中次数}}{S\cdot K}$$
得到的是：**该专家被选中的全局占比**
$$\sum_e ce_{\text{raw}} = 1$$

### ② 除以 $\boldsymbol{\dfrac{SK}{E}}$ 之后（带E）
$$ce_{\text{norm}}
= \frac{\text{专家}e\text{选中次数}}{\;\dfrac{SK}{E}\;}$$

👉 这一步的含义：
**用「当前专家实际选中次数」 / 「专家完全均分下的理想选中次数」**

---

## 5 物理意义（核心）
$$ce_{\text{norm}}[e]
= \frac{\text{专家}e\text{实际负载}}{\text{全局均匀理想负载}}$$

- $ce_{\text{norm}}[e] = 1$ → 负载**完美均衡**
- $ce_{\text{norm}}[e] > 1$ → 该专家**过载、被选太多**
- $ce_{\text{norm}}[e] < 1$ → 该专家**闲置、选得太少**

✅ **只有除以 $E$，才能把指标归一到「以均匀分布为基准」的相对负载**
✅ 不分专家总数多少、不分序列长短，负载数值**跨配置可比**

---

## 6 结合你后面的损失一起理解
```python
aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(dim=1).mean() * alpha
```
这是经典 **MoE 负载均衡正则项**（Switch / Qwen-MoE 同款）：
- $ce$：**归一化后的专家负载倍率**（靠除以E实现）
- $scores$：路由给每个专家的权重分数

损失本质是在约束：
> 路由分数高的专家，实际负载倍率也要匹配，
> 强迫所有专家负载倍率尽量靠近 1，全局均衡。

---

## 7 一句话总结
1. $\boldsymbol{SK}$：归一掉**序列长度、每个token选几个专家**的影响；
2. $\boldsymbol{\div E}$：引入**专家总数**，算出**相对于完全均匀分配的负载倍率**；
3. 没有除以 `num_routed_experts`，就**得不到标准负载倍率**，负载均衡loss会完全失效，专家极易崩塌、两极分化。


# `ce` 代码段讲解
逐行、逐算子**通俗+数学+维度**完整拆解，结合 MoE 辅助负载损失场景，讲清楚每一步作用。

---

## 完整代码
```python
ce.scatter_add_(
    1, 
    topk_idx_for_aux_loss, 
    torch.ones(batch_size, seq_len * aux_topk, device=hidden_states.device),
).div_(seq_len * aux_topk / self.num_routed_experts)
```

---

## 1. 先明确变量 & 维度
设：
- `batch_size`：批次大小 $B$
- `seq_len`：序列长度 $L$
- `aux_topk`：aux 路由每个token选的专家数 $K$
- `num_routed_experts`：路由专家总数 $E$

---

## 2. 核心算子：`scatter_add_(dim, index, src)`
### 作用
**按索引累加求和**：
以 `index` 为下标，把 `src` 里的值，累加到 `ce` 的对应位置。
- `dim=1`：在**第1维**做散列累加
- 这是 MoE 统计「每个专家被多少个 token 选中」的标准写法

---

## 3. 逐参数拆解

### ① `ce`
- 形状：$[B,\ E]$
- 含义：**每个批次、每个专家的被选中计数容器**
- 初始一般是全0张量，用来累计专家被调用次数

### ② `dim = 1`
维度1 对应「专家维度」，也就是：
> 按专家索引位置，往对应专家位置累加计数

### ③ `topk_idx_for_aux_loss`
- shape：$[B,\ L\times K]$
- 内容：**所有 token 选出的 aux 专家索引**
  把 `[B, L, K]` 展平成 `[B, L*K]`，方便 scatter

### ④ 第三个参数：全1张量
```python
torch.ones(batch_size, seq_len * aux_topk, ...)
```
shape 和 `topk_idx_for_aux_loss` 完全一致
- 每选中一个专家，就**贡献 1 次计数**
- 等价：只要token选了这个专家，该专家计数+1

---

## 4. scatter_add_ 整体效果
```
ce[batch, expert_idx] += 1
```
遍历所有样本、所有token的aux选中专家，
**统计出：当前批次里，每个专家一共被选中多少次**

执行完 scatter_add_ 后：
$$ce[b,e] = \text{第}b\text{个样本中，专家}e\text{被选中的总次数}$$

---

## 5. 后半段：`.div_(seq_len * aux_topk / self.num_routed_experts)`
原地除法归一化，先写表达式：
$$ce = ce \div \left( \frac{L \cdot K}{E} \right)$$
等价变形：
$$ce = ce \cdot \frac{E}{L \cdot K}$$

#### 分母含义
- $L\cdot K$：整个batch里，**所有token的aux专家选择总次数**
- $E$：专家总数

#### 归一化目的
把「绝对选中次数」转为**专家负载占比**，
让负载数值和 batch/序列长度/topk 解耦，
方便后续算 **aux_loss（负载均衡损失）**。

---

## 6. 结合你上一行代码联动理解
你上一行：
```python
load = F.one_hot(...).mean(0)
```
是用 onehot+mean 算专家平均负载；

**本行是工业界更高效实现**：
用 `scatter_add_` 稀疏累加计数 → 再归一化，
显存/速度远优于 one_hot（专家一多 onehot 会爆炸）。

---

## 7. 一句话极简总结
1. 用 `scatter_add_` 稀疏统计**每个专家被 aux 路由选中的总次数**；
2. 用全局选择数 & 专家总数做归一化；
3. 最终得到**各专家标准化负载**，用于 MoE 辅助负载均衡损失，防止专家坍缩、冷热不均。

---

## 补充小例子（方便直观理解）
$E=4,\ L=2,\ K=2$
某个样本所有token选的专家：`[0,1,1,3]`
ones 就是 `[1,1,1,1]`
scatter_add 后：
expert0=1，expert1=2，expert3=1，expert2=0
再做归一化：
$$L\cdot K = 4,\quad \dfrac{E}{L\cdot K} = \dfrac{4}{4}=1$$
负载保持计数比例，直接用于均衡损失计算。




# 批级辅助损失计算（seq_aux = False）

对照公式和之前的代码，把每一步一一对应起来，就能看明白这段代码是怎么实现批级辅助损失的。

---

## 一、先把公式和代码变量对应上
先把所有符号翻译成代码里的变量，这样对照起来会非常清晰：

| 公式符号 | 含义 | 代码中的对应变量 |
| :--- | :--- | :--- |
| $E$ | 专家总数 | `self.n_routed_experts` |
| $N$ | 批次样本数 | 隐含在 `topk_idx_for_aux_loss` 中 |
| $k$ | 每个样本选的专家数（top-k） | 隐含在 `topk_idx_for_aux_loss` 中 |
| $m_{i,e}$ | 第$i$个样本是否选了专家$e$（one-hot） | `mask_ce`（one-hot 结果） |
| $f_e$ | 专家$e$的全局平均选择率 | `ce`（`mask_ce.float().mean(0)`） |
| $\hat{f}_e$ | 标准化后的相对负载因子 | `fi`（`ce * self.n_routed_experts`） |
| $s_{i,e}$ | 第$i$个样本对专家$e$的路由分数 | `scores_for_aux` |
| $p_e$ | 专家$e$的全局平均分数 | `Pi`（`scores_for_aux.mean(0)`） |
| $\alpha$ | 辅助损失权重系数 | `self.alpha` |
| $\mathcal{L}_{\text{aux}}^{\text{batch}}$ | 最终批级辅助损失 | `aux_loss` |

---

## 二、逐步骤对照公式与代码实现

### 步骤 1：计算专家 $e$ 的全局平均选择率 $f_e$
公式：
$$f_e = \frac{1}{N \cdot k} \sum_{i=1}^{N \cdot k} m_{i,e}$$
代码实现：
```python
mask_ce = F.one_hot(
    topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts
)
ce = mask_ce.float().mean(0)
```
- `topk_idx_for_aux_loss.view(-1)`：把 `(N, k)` 的 top-k 索引展平成 `(N·k,)`，对应公式里的 $N \cdot k$ 个样本-专家选择。
- `F.one_hot(...)`：生成 one-hot 编码，就是公式里的 $m_{i,e}$，表示第 $i$ 个选择里，是否选了专家 $e$。
- `.float().mean(0)`：在 `N·k` 维度上求均值，就是公式里的 $\frac{1}{N \cdot k} \sum$，直接得到 $f_e$。

---

### 步骤 2：标准化为“相对负载因子” $\hat{f}_e$
公式：
$$\hat{f}_e = f_e \cdot E$$
代码实现：
```python
fi = ce * self.n_routed_experts
```
- 这里 `ce` 就是上一步的 $f_e$，`self.n_routed_experts` 就是 $E$，直接完成标准化。
- 当专家负载均衡时，$f_e = 1/E$，此时 $\hat{f}_e = 1$，和公式说明完全一致。

---

### 步骤 3：计算专家 $e$ 的全局平均分数 $p_e$
公式：
$$p_e = \frac{1}{N} \sum_{i=1}^{N} s_{i,e}$$
代码实现：
```python
Pi = scores_for_aux.mean(0)
```
- `scores_for_aux` 就是每个样本对所有专家的路由分数 $s_{i,e}$，形状是 `(N, E)`。
- `.mean(0)`：在 `N` 个样本维度上求均值，就是公式里的 $\frac{1}{N} \sum$，直接得到 $p_e$。

---

### 步骤 4：计算批级辅助损失 $\mathcal{L}_{\text{aux}}^{\text{batch}}$
公式：
$$\mathcal{L}_{\text{aux}}^{\text{batch}} = \alpha \cdot \sum_{e=1}^{E} \hat{f}_e \cdot p_e$$
代码实现：
```python
aux_loss = (Pi * fi).sum() * self.alpha
```
- `Pi * fi`：对应公式里的 $\hat{f}_e \cdot p_e$，逐元素相乘。
- `.sum()`：对所有专家 $e$ 求和，即 $\sum_{e=1}^{E}$。
- `* self.alpha`：乘以权重系数 $\alpha$，完成最终损失计算。

---

## 三、直观解释：代码里的损失是怎么让专家均衡的？
图片里的直观解释，完全对应这段代码的逻辑：
1.  **损失项 $\hat{f}_e \cdot p_e$ 的作用**：
    - 如果专家 $e$ 负载很高（$\hat{f}_e > 1$，说明被选得太多），损失会鼓励模型降低对它的路由分数 $p_e$。
    - 如果专家 $e$ 负载很低（$\hat{f}_e < 1$，说明被选得太少），损失会鼓励模型提高对它的路由分数 $p_e$。
2.  **梯度反传的效果**：
    门控网络（gating network）会根据这个损失调整每个专家的路由分数，让所有专家的 $\hat{f}_e$ 都趋近于 1，最终实现负载均衡。
3.  **计算高效**：
    整个过程都是在批次维度上做均值和矩阵运算，没有复杂的序列级操作，适合大规模训练，这也是 Switch Transformer 采用这种方式的原因。

---

## 四、总结：这段代码就是公式的直接实现

- 用 one-hot + 均值 计算 $f_e$（实际选择率）
- 乘以专家数 $E$ 得到 $\hat{f}_e$（相对负载因子）
- 用路由分数的均值计算 $p_e$（模型分配的平均分数）
- 逐元素相乘、求和、乘以权重 $\alpha$ 得到最终损失。

---
