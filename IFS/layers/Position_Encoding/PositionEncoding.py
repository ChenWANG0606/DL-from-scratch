import math
import torch
from torch import nn


class PositionEncoding(nn.Module):
    """
    为序列中的 token 注入位置信息。
    这里的位置编码没有可学习参数，而是由正弦和余弦函数直接构造得到。
    """
    def __init__(self, embed_dim, dropout=0.1, max_len=5000):
        """
        构造位置编码层。

        输入参数：
         - embed_dim: 词向量嵌入维度
         - dropout: dropout 概率
         - max_len: 输入序列允许的最大长度
        """

        super().__init__()
        self.dropout = nn.Dropout(dropout)
        assert embed_dim%2 == 0

        # 先创建一个 batch 维为 1 的位置编码张量，后续可自动广播到整个 batch。
        pe = torch.zeros(1, max_len, embed_dim)

        # 根据论文中的公式构造位置编码矩阵。
        # ------------------------------------------------------------------------------------------------------------------------------
        position = torch.arange(max_len).unsqueeze(-1) 
        div_term = torch.exp(torch.arange(0, embed_dim, 2)*(-math.log(10000)/embed_dim))
        pe[0, :, 0::2] = torch.sin(position*div_term)
        pe[0, :, 1::2] = torch.cos(position*div_term)
        # ------------------------------------------------------------------------------------------------------------------------------

        # 将位置编码注册为 buffer，保存模型时会一并保存，但不会参与梯度更新。
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        将位置编码逐元素加到输入序列上。

        输入：
         - x: 输入序列，形状为 (N, S, D)
              其中 N 为 batch size，S 为序列长度，D 为嵌入维度
        返回：
         - output: 加上位置编码后的结果，形状为 (N, S, D)
        """
        if x.dim() != 3:
            raise ValueError(f"x 的形状应为 (N, S, D)，当前得到的是 {tuple(x.shape)}")

        # 按当前序列长度截取位置编码，并与输入相加，最后施加 dropout。
        # ------------------------------------------------------------------------------------------------------------------------------
        _ , S, E = x.shape
        output = self.dropout(self.pe[:,:S,:]+x)
        # ------------------------------------------------------------------------------------------------------------------------------

        return output


class RoPE(nn.Module):
    """
    Rotary Position Embedding.
    与 PositionEncoding 不同，RoPE 不直接加到输入上，
    而是在注意力计算前对 q 和 k 做旋转变换。
    """
    def __init__(self, head_dim, max_len=5000, base=10000):
        """
        构造 RoPE 层。

        输入参数：
         - head_dim: 每个注意力头的维度
         - max_len: 输入序列允许的最大长度
         - base: 频率基数，通常取 10000
        """
        super().__init__()
        assert head_dim % 2 == 0

        self.head_dim = head_dim
        self.max_len = max_len
        self.base = base

        cos = torch.zeros(1, 1, max_len, head_dim)
        sin = torch.zeros(1, 1, max_len, head_dim)

        # 根据 RoPE 公式预先构造 cos 和 sin 查找表。
        # q / k 的形状通常为 (N, H, S, D)，这里预计算后方便按序列长度切片。
        # ------------------------------------------------------------------------------------------------------------------------------
        position = torch.arange(max_len).unsqueeze_(-1)
        theta = torch.exp(torch.arange(0, head_dim, 2) * (-math.log(base)/head_dim))
        cos[0,0,:,:] = torch.repeat_interleave(torch.cos(position * theta),2,dim = -1)
        sin[0,0,:,:] = torch.repeat_interleave(torch.sin(position * theta),2,dim = -1)
        # ------------------------------------------------------------------------------------------------------------------------------
        self.register_buffer("cos_cached", cos)
        self.register_buffer("sin_cached", sin)

    def rotate_half(self, x):
        """
        将最后一维按两个一组进行旋转。

        输入：
         - x: 形状为 (..., D)
        返回：
         - rotated_x: 形状仍为 (..., D)
        """
        # 将最后一维拆成两部分，并按 RoPE 的形式重排。
        # ------------------------------------------------------------------------------------------------------------------------------
        rotated_x = torch.zeros_like(x)
        rotated_x[:,:,:,0::2] = -x[:,:,:,1::2]
        rotated_x[:,:,:,1::2] = x[:,:,:,0::2]
        # ------------------------------------------------------------------------------------------------------------------------------
        return rotated_x

    def forward(self, q, k):
        """
        对 q 和 k 施加旋转位置编码。

        输入：
         - q: query，形状为 (N, H, S, D)
         - k: key，形状为 (N, H, S, D)
        返回：
         - q_rotated: 施加 RoPE 后的 q
         - k_rotated: 施加 RoPE 后的 k
        """
        if q.dim() != 4 or k.dim() != 4:
            raise ValueError("q 和 k 的形状都应为 (N, H, S, D)")
        if q.shape != k.shape:
            raise ValueError("q 和 k 的形状必须一致")

        _, _, seq_len, dim = q.shape
        if dim != self.head_dim:
            raise ValueError(
                f"最后一维应为 {self.head_dim}，当前得到的是 {dim}"
            )

        # 按当前序列长度截取 cos / sin，并对 q、k 做旋转。
        # ------------------------------------------------------------------------------------------------------------------------------
        cos = self.cos_cached[:,:,:seq_len,:]
        sin = self.sin_cached[:,:,:seq_len,:]

        q_rotated = q*cos+self.rotate_half(q)*sin
        k_rotated = k*cos+self.rotate_half(k)*sin

        # ------------------------------------------------------------------------------------------------------------------------------
        return q_rotated, k_rotated


def _build_expected_rope(x, base=10000):
    _, _, seq_len, head_dim = x.shape
    expected = torch.zeros_like(x)

    for pos in range(seq_len):
        for i in range(0, head_dim, 2):
            angle = pos / (base ** (i / head_dim))
            cos_theta = math.cos(angle)
            sin_theta = math.sin(angle)

            expected[..., pos, i] = (
                x[..., pos, i] * cos_theta - x[..., pos, i + 1] * sin_theta
            )
            expected[..., pos, i + 1] = (
                x[..., pos, i] * sin_theta + x[..., pos, i + 1] * cos_theta
            )

    return expected


def main():
    head_dim = 8
    seq_len = 6
    batch_size = 2
    num_heads = 3

    model = RoPE(head_dim=head_dim, max_len=seq_len)
    model.eval()

    q = torch.randn(batch_size, num_heads, seq_len, head_dim)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim)
    q_output, k_output = model(q, k)
    q_expected = _build_expected_rope(q, base=model.base)
    k_expected = _build_expected_rope(k, base=model.base)

    assert q_output.shape == (batch_size, num_heads, seq_len, head_dim), (
        f"q 输出形状不符合预期: {tuple(q_output.shape)}"
    )
    assert k_output.shape == (batch_size, num_heads, seq_len, head_dim), (
        f"k 输出形状不符合预期: {tuple(k_output.shape)}"
    )
    assert torch.allclose(q_output, q_expected, atol=1e-6), (
        "RoPE 的 q 输出与旋转公式不一致。"
    )
    assert torch.allclose(k_output, k_expected, atol=1e-6), (
        "RoPE 的 k 输出与旋转公式不一致。"
    )
    assert torch.allclose(q_output[:, :, 0, :], q[:, :, 0, :], atol=1e-6), (
        "位置 0 的 q 旋转结果应与输入一致。"
    )
    assert torch.allclose(k_output[:, :, 0, :], k[:, :, 0, :], atol=1e-6), (
        "位置 0 的 k 旋转结果应与输入一致。"
    )

    print("RoPE 验证通过。")
    print("q 输出形状:", tuple(q_output.shape))
    print("k 输出形状:", tuple(k_output.shape))
    print("q 第一个位置:", q_output[0, 0, 0])
    print("q 第二个位置:", q_output[0, 0, 1])


if __name__ == "__main__":
    main()
