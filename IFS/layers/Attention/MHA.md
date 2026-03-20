MultiHead Attention

#### Query, Key, Value

In Transformers, we perform self-attention, which means that the values, keys and query are derived from the input $X \in \mathbb{R}^{\ell \times d_1}$, where $\ell$ is our sequence length. Specifically, we learn parameter matrices $V_i,K_i,Q_i \in \mathbb{R}^{d_1\times d/h}$ to map our input $X$ as follows:

$$
\begin{aligned}
v_i = V_iX\ \ i \in \{1,\dots,h\}\\
k_i = K_iX\ \ i \in \{1,\dots,h\}\\
q_i = Q_iX\ \ i \in \{1,\dots,h\}
\end{aligned}
$$

where $i$ refers to the $i$-th head and $h$ is the number of heads.


#### Multi-Headed Scaled Dot-Product Attention
In the case of multi-headed attention, we learn a parameter matrix for each head, which gives the model more expressivity to attend to different parts of the input. Let $Y_i$ be the attention output of head $i$. Thus we learn individual matrices $Q_i$, $K_i$ and $V_i$. To keep our overall computation the same as the single-headed case, we choose $Q_i \in \mathbb{R}^{d\times d/h}$, $K_i \in \mathbb{R}^{d\times d/h}$ and $V_i \in \mathbb{R}^{d\times d/h}$. Adding in a scaling term $\frac{1}{\sqrt{d/h}}$ to our simple dot-product attention above, we have

$$ 
A_i = \text{softmax}\bigg(\frac{(XQ_i)(XK_i)^\top}{\sqrt{d/h}}\bigg)
$$

Now we have got our attention $A_i$, and each head's output could be calculated using the following formula.

$$
Y_i = A_i(XV_i)
$$

where $Y_i\in\mathbb{R}^{\ell \times d/h}$, where $\ell$ is our sequence length.


In our implementation, we apply dropout to the attention weights (though in practice it could be used at any step):
$$ 
Y_i = \text{dropout}\bigg(A_i\bigg)(XV_i)
$$

Finally, then the output of the self-attention is a linear transformation of the concatenation of the heads:

$$
Y = [Y_1;\dots;Y_h]W
$$

where $W \in\mathbb{R}^{d\times d}$ and $[Y_1;\dots;Y_h]\in\mathbb{R}^{\ell \times d}$.
