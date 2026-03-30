# RoPE+YaRN
## RoPE（Rotary Positional Embedding）旋转位置编码
* 待学习RoPE（Rotary Positional Embedding）通过一种旋转变换，将位置编码转换为旋转矩阵  


- 初始化基准RoPE频率：
$$
freqs = \frac{1}{{rope\_base}^{\frac{2i}{dim}}}
$$
其中，$rope\_base$ 是一个超参数，$i$ 是位置编码的维度索引，$dim$ 是位置编码的总维度。  
- 波长边界计算（当推理长度大于训练长度）：
  - 频率映射到波长
  $$\lambda = 2\pi*\frac{1}{freqs} = 2\pi*{rope\_base}^{\frac{2i}{dim}}$$
  - 设定的训练文本长度为 $L$ ，与其长度对应波长 $\lambda_i$ （维度索引）的比例系数为$b$
  $$\frac{L}{\lambda_i} = b$$
  - 将设定的高、低频分界点的比例系数 $b$ 带入公式，可计算频率分界点的维度索引 $i$ ：
  $$\lambda_i = \frac{L}{b} = 2\pi*{rope\_base}^{\frac{2i}{dim}}$$
  $${rope\_base}^{\frac{2i}{dim}} = \frac{L}{2\pi*b}$$
  $$i = \frac{dim*\ln{\frac{L}{2\pi*b}}}{2*\ln{rope\_base}}$$  
  - 计算混合因子（用波长计算，即频率的倒数）：
  $$令：f = (\frac{dim}{2} - i_{low})/(i_{high} - i_{low})$$
  $$
  ramp = \left\{ \begin{array}{rcl} 
         0 &, & f \leq 0 \\
         \\
         f &, & 0 < f < 1 \\
         \\
         1 &, & f \geq 1 
         \end{array}\right.
  $$






## YaRN（Yet Another Relative Positional Embedding）相对位置编码
* 待学习YaRN（Yet Another Relative Positional Embedding）通过一种相对位置编码，将位置编码转换为相对位置编码

