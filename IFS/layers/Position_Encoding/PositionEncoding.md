### 位置编码
#### 正弦位置编码

虽然 Transformer 可以方便地关注输入中的任意位置，但注意力机制本身并不包含顺序信息。然而在很多任务中，尤其是自然语言处理任务里，词语的相对顺序非常重要。为了解决这个问题，作者会在每个 token 的词向量上加入位置编码。

定义位置编码矩阵 $P \in \mathbb{R}^{l\times d}$，其中 $P_{ij}$ 的计算方式如下：
$$
\begin{cases}
PE_{(pos,\,2i)} = \sin\left(\frac{pos}{10000^{\frac{2i}{d}}}\right)\\
PE_{(pos,\,2i+1)} = \cos\left(\frac{pos}{10000^{\frac{2i}{d}}}\right)
\end{cases}
$$
$$
\begin{cases}
\text{sin}\left(i \cdot 10000^{-\frac{j}{d}}\right) & \text{当 } j \text{ 为偶数时} \\
\text{cos}\left(i \cdot 10000^{-\frac{(j-1)}{d}}\right) & \text{当 } j \text{ 为奇数时} \\
\end{cases}
$$

因此，送入网络的并不是原始输入 $X \in \mathbb{R}^{l\times d}$，而是加上位置编码后的结果 $X + P$。

#### 旋转位置编码（RoPE）

正弦位置编码是把位置信息直接加到输入表示上，而 RoPE（Rotary Position Embedding）则是把位置信息作用到注意力中的 $Q$ 和 $K$ 上。它的核心思想是：将每两个维度看作一个二维平面中的坐标，再按照当前位置对应的角度做旋转。

对于第 $m$ 个位置、最后一维中的第 $2i$ 和第 $2i+1$ 维，先定义旋转角频率：

$$
\theta_i = 10000^{-\frac{2i}{d}}
$$

然后把向量的这两个维度组成一个二维向量，并做旋转：

$$
\begin{pmatrix}
x_{2i}^{(m)\prime} \\
x_{2i+1}^{(m)\prime}
\end{pmatrix}
$$

=

$$
\begin{pmatrix}
\cos(m\theta_i) & -\sin(m\theta_i) \\
\sin(m\theta_i) & \cos(m\theta_i)
\end{pmatrix}
\begin{pmatrix}
x_{2i}^{(m)} \\
x_{2i+1}^{(m)}
\end{pmatrix}
$$

写成逐元素形式就是：
$$
\begin{cases}
x_{2i}^{(m)\prime} = x_{2i}^{(m)} \cos(m\theta_i) - x_{2i+1}^{(m)} \sin(m\theta_i) \\
x_{2i+1}^{(m)\prime} = x_{2i}^{(m)} \sin(m\theta_i) + x_{2i+1}^{(m)} \cos(m\theta_i)
\end{cases}
$$

在实现时，通常不会直接旋转输入 $X$，而是在线性映射得到 $Q$ 和 $K$ 之后，对它们应用同样的旋转：
$$
Q' = \text{RoPE}(Q), \quad K' = \text{RoPE}(K)
$$
然后再计算注意力分数：
$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q'K'^{T}}{\sqrt{d}}\right)V
$$

RoPE 的一个重要优点是：它能够把相对位置信息自然编码到注意力分数中，因此在长序列建模里经常比直接相加的位置编码更有效。

总结来说：

- 正弦位置编码：直接加到输入 $X$ 上，得到 $X + P$
- RoPE：不直接修改输入，而是旋转注意力中的 $Q$ 和 $K$
- 使用 RoPE 时，通常要求每个 attention head 的维度是偶数，这样才能两两配对做旋转
