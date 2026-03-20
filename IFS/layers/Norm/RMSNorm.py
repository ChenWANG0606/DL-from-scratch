import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """
    支持任意维度输入的 RMSNorm，并支持训练（gamma 为可学习参数）。

    输入:
        x: (..., *normalized_shape)

    参数:
        normalized_shape: int 或 tuple，表示需要做归一化的最后几个维度
        eps: 数值稳定项

    与 LayerNorm 的区别：
        LayerNorm: (x - mean) / std
        RMSNorm:   x / rms
    """

    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()

        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)

        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps

        # 可学习缩放参数
        self.gamma = nn.Parameter(torch.ones(self.normalized_shape))

    def forward(self, x):
        """
        x: shape (..., *normalized_shape)
        """

        assert x.shape[-len(self.normalized_shape):] == self.normalized_shape, (
            f"expect last dims {self.normalized_shape}, got {x.shape}"
        )

        # 需要归一化的维度
        dims = tuple(range(-len(self.normalized_shape), 0))

        # 计算 mean square
        mean_square = (x ** 2).mean(dim=dims, keepdim=True)

        # 计算 rms
        rms = torch.sqrt(mean_square + self.eps)

        # 归一化
        x_hat = x / rms

        # 缩放
        y = self.gamma * x_hat

        return y


# =========================
# 参考输入 / 目标输出（方便你手算验证）
# =========================
def demo_case():
    """
    一个非常小的例子，便于你理解每一步。

    输入:
        [[1, 1],
         [2, 2]]

    对每一行做 RMSNorm：
    第 1 行 mean_square = (1^2 + 1^2) / 2 = 1
    第 1 行 rms = sqrt(1) = 1

    第 2 行 mean_square = (2^2 + 2^2) / 2 = 4
    第 2 行 rms = sqrt(4) = 2

    标准化后约为:
        [[1, 1],
         [1, 1]]
    （严格来说会因为 eps 稍微小一点点）

    因为初始 gamma=1，所以输出 y ≈ x_hat
    """
    x = torch.tensor([
        [1.0, 1.0],
        [2.0, 2.0],
    ])

    target = torch.tensor([
        [1.0, 1.0],
        [1.0, 1.0],
    ])

    return x, target


# =========================
# 测试代码
# 你补完 TODO 后，直接运行本文件即可
# =========================
def test_shape():
    rms_norm = RMSNorm(normalized_shape=3)
    x = torch.randn(8, 3)
    y = rms_norm.forward(x)
    assert y.shape == x.shape


def test_row_rms_normalization():
    """
    当 gamma=1 时，输出应接近“每行的均方值为 1”。
    注意：RMSNorm 不保证每行均值为 0。
    """
    rms_norm = RMSNorm(normalized_shape=4)
    x = torch.randn(32, 4)
    y = rms_norm.forward(x)

    row_mean_square = (y ** 2).mean(dim=1)

    assert torch.allclose(row_mean_square, torch.ones(32), atol=1e-4), row_mean_square


def test_demo_case_close():
    rms_norm = RMSNorm(normalized_shape=2, eps=1e-5)
    x, target = demo_case()
    y = rms_norm.forward(x)

    # 由于 eps 存在，数值不会严格等于 1，所以这里放宽一点
    assert torch.allclose(y, target, atol=1e-3), y


def test_affine_transform():
    rms_norm = RMSNorm(normalized_shape=2, eps=1e-5)
    rms_norm.gamma = torch.tensor([[2.0, 0.5]])

    x, _ = demo_case()
    y = rms_norm.forward(x)

    expected = torch.tensor([
        [2.0, 0.5],
        [2.0, 0.5],
    ])
    assert torch.allclose(y, expected, atol=1e-3), y


def run_all_tests():
    test_shape()
    test_row_rms_normalization()
    test_demo_case_close()
    test_affine_transform()
    print("All tests passed! 你已经完成了 RMSNorm 的基础实现。")


if __name__ == "__main__":
    print("===== RMSNorm 填空练习 =====")
    x, target = demo_case()
    print("示例输入 x:\n", x)
    print("目标输出（近似）:\n", target)
    print()

    run_all_tests()
