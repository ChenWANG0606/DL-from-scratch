import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """
    支持任意维度的 LayerNorm 实现，并且支持训练（gamma / beta 为可学习参数）。

    输入:
        x: (..., *normalized_shape)

    参数:
        normalized_shape: int 或 tuple，表示需要做归一化的最后几个维度
        eps: 数值稳定项

    例如:
        x.shape = (N, D)           normalized_shape=D
        x.shape = (N, T, D)        normalized_shape=D
        x.shape = (N, C, H, W)     normalized_shape=(H, W)
    """

    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()

        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)

        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps

        # 可学习参数
        self.gamma = nn.Parameter(torch.ones(self.normalized_shape))
        self.beta = nn.Parameter(torch.zeros(self.normalized_shape))

    def forward(self, x):
        """
        x: shape (..., *normalized_shape)
        """

        assert x.shape[-len(self.normalized_shape):] == self.normalized_shape, (
            f"expect last dims {self.normalized_shape}, got {x.shape}"
        )

        # 需要归一化的维度
        dims = tuple(range(-len(self.normalized_shape), 0))

        # 均值
        mean = x.mean(dim=dims, keepdim=True)

        # 方差（总体方差）
        var = x.var(dim=dims, keepdim=True, unbiased=False)

        # 标准化
        x_hat = (x - mean) / torch.sqrt(var + self.eps)

        # 仿射变换（自动广播）
        y = self.gamma * x_hat + self.beta

        return y


# =========================
# 参考输入 / 目标输出（方便你手算验证）
# =========================
def demo_case():
    """
    一个非常小的例子，便于你理解每一步。

    输入:
        [[1, 3],
         [2, 4]]

    对每一行做 LN：
    第 1 行均值 = 2, 方差 = 1
    第 2 行均值 = 3, 方差 = 1

    标准化后约为:
        [[-1,  1],
         [-1,  1]]
    （严格来说会因为 eps 稍微小一点点）

    因为初始 gamma=1, beta=0，所以输出 y ≈ x_hat
    """
    x = torch.tensor([
        [1.0, 3.0],
        [2.0, 4.0],
    ])

    target = torch.tensor([
        [-1.0, 1.0],
        [-1.0, 1.0],
    ])

    return x, target


# =========================
# 测试代码
# 你补完 TODO 后，直接运行本文件即可
# =========================
def test_shape():
    ln = LayerNorm(normalized_shape=3)
    x = torch.randn(8, 3)
    y = ln.forward(x)
    assert y.shape == x.shape


def test_row_normalization():
    """
    当 gamma=1, beta=0 时，输出应接近“每行均值为 0、方差为 1”。
    """
    ln = LayerNorm(normalized_shape=4)
    x = torch.randn(32, 4)
    y = ln.forward(x)

    row_mean = y.mean(dim=1)
    row_var = y.var(dim=1, unbiased=False)

    assert torch.allclose(row_mean, torch.zeros(32), atol=1e-6), row_mean
    assert torch.allclose(row_var, torch.ones(32), atol=1e-4), row_var


def test_demo_case_close():
    ln = LayerNorm(normalized_shape=2, eps=1e-5)
    x, target = demo_case()
    y = ln.forward(x)

    # 由于 eps 存在，数值不会严格等于 ±1，所以这里放宽一点
    assert torch.allclose(y, target, atol=1e-3), y


def test_affine_transform():
    ln = LayerNorm(normalized_shape=2, eps=1e-5)
    ln.gamma = torch.tensor([[2.0, 0.5]])
    ln.beta = torch.tensor([[1.0, -1.0]])

    x, _ = demo_case()
    y = ln.forward(x)

    expected = torch.tensor([
        [-1.0, -0.5],
        [-1.0, -0.5],
    ])
    assert torch.allclose(y, expected, atol=1e-3), y


def run_all_tests():
    test_shape()
    test_row_normalization()
    test_demo_case_close()
    test_affine_transform()
    print("All tests passed! 你已经完成了 LayerNorm 的基础实现。")


if __name__ == "__main__":
    print("===== LayerNorm 填空练习 =====")
    x, target = demo_case()
    print("示例输入 x:\n", x)
    print("目标输出（近似）:\n", target)
    print()

    run_all_tests()
