import torch
import torch.nn as nn

class BatchNorm1d(nn.Module):
    """
    支持训练的 BatchNorm1d 实现。

    输入:
        x: (N, D)

    参数:
        num_features: 特征维度 D
        eps: 数值稳定项
        momentum: running statistics 的动量

    训练时：使用 batch statistics
    推理时：使用 running statistics
    """

    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()

        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        # 可学习参数
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))

        # running statistics (buffer 不参与梯度更新)
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def forward(self, x, training=True):
        """
        x: (N, D)
        """

        assert x.ndim == 2, f"expect 2D input, got shape={x.shape}"
        assert x.shape[1] == self.num_features, (
            f"expect feature dim = {self.num_features}, got {x.shape[1]}"
        )

        if training:
            # batch statistics
            batch_mean = x.mean(dim=0)
            batch_var = x.var(dim=0, unbiased=False)

            # normalize
            x_hat = (x - batch_mean) / torch.sqrt(batch_var + self.eps)

            # update running stats
            self.running_mean = (
                self.momentum * self.running_mean
                + (1 - self.momentum) * batch_mean
            )

            self.running_var = (
                self.momentum * self.running_var
                + (1 - self.momentum) * batch_var
            )

        else:
            # inference
            x_hat = (x - self.running_mean) / torch.sqrt(self.running_var + self.eps)

        # affine transform (broadcast)
        y = self.gamma * x_hat + self.beta

        return y


# =========================
# 参考输入 / 目标输出（方便你手算验证）
# =========================
def demo_case():
    """
    一个非常小的例子，便于你理解每一步。

    输入:
        [[1, 2],
         [3, 4]]

    对每一列做 BN：
    第 1 列均值 = 2, 方差 = 1
    第 2 列均值 = 3, 方差 = 1

    标准化后约为:
        [[-1, -1],
         [ 1,  1]]
    （严格来说会因为 eps 稍微小一点点）

    因为初始 gamma=1, beta=0，所以输出 y ≈ x_hat
    """
    x = torch.tensor([
        [1.0, 2.0],
        [3.0, 4.0],
    ])

    target = torch.tensor([
        [-1.0, -1.0],
        [1.0, 1.0],
    ])

    return x, target


# =========================
# 测试代码
# 你补完 TODO 后，直接运行本文件即可
# =========================
def test_shape():
    bn = BatchNorm1d(num_features=3)
    x = torch.randn(8, 3)
    y = bn.forward(x, training=True)
    assert y.shape == x.shape


def test_training_normalization():
    """
    当 gamma=1, beta=0 时，训练态输出应接近“每列均值为 0、方差为 1”。
    """
    bn = BatchNorm1d(num_features=4)
    x = torch.randn(32, 4)
    y = bn.forward(x, training=True)

    col_mean = y.mean(dim=0)
    col_var = y.var(dim=0, unbiased=False)

    assert torch.allclose(col_mean, torch.zeros(4), atol=1e-6), col_mean
    assert torch.allclose(col_var, torch.ones(4), atol=1e-4), col_var


def test_demo_case_close():
    bn = BatchNorm1d(num_features=2, eps=1e-5)
    x, target = demo_case()
    y = bn.forward(x, training=True)

    # 由于 eps 存在，数值不会严格等于 ±1，所以这里放宽一点
    assert torch.allclose(y, target, atol=1e-3), y


def test_running_stats_updated():
    bn = BatchNorm1d(num_features=2, momentum=0.1)
    x = torch.tensor([
        [1.0, 2.0],
        [3.0, 4.0],
    ])
    _ = bn.forward(x, training=True)

    # 当前 batch: mean=[[2,3]], var=[[1,1]]
    # 初始 running_mean=[[0,0]], running_var=[[1,1]]
    # 更新后:
    # running_mean = 0.1 * [0,0] + 0.9 * [2,3] = [1.8, 2.7]
    # running_var  = 0.1 * [1,1] + 0.9 * [1,1] = [1,1]
    expected_running_mean = torch.tensor([[1.8, 2.7]])
    expected_running_var = torch.tensor([[1.0, 1.0]])

    assert torch.allclose(bn.running_mean, expected_running_mean, atol=1e-6), bn.running_mean
    assert torch.allclose(bn.running_var, expected_running_var, atol=1e-6), bn.running_var


def test_inference_uses_running_stats():
    bn = BatchNorm1d(num_features=2, momentum=0.1)

    # 先喂一个 batch，让 running stats 更新
    x_train = torch.tensor([
        [1.0, 2.0],
        [3.0, 4.0],
    ])
    _ = bn.forward(x_train, training=True)

    # 推理时应使用 running_mean=[1.8, 2.7], running_var=[1,1]
    x_test = torch.tensor([
        [1.8, 2.7],
        [2.8, 3.7],
    ])
    y = bn.forward(x_test, training=False)

    # 第一行正好等于 running mean，归一化后应该接近 0
    # 第二行比 running mean 每列都大 1，归一化后应该接近 1
    expected = torch.tensor([
        [0.0, 0.0],
        [1.0, 1.0],
    ])
    assert torch.allclose(y, expected, atol=1e-3), y


def run_all_tests():
    test_shape()
    test_training_normalization()
    test_demo_case_close()
    test_running_stats_updated()
    test_inference_uses_running_stats()
    print("All tests passed! 你已经完成了 BatchNorm1d 的基础实现。")


if __name__ == "__main__":

    print("===== BatchNorm 填空练习 =====")
    x, target = demo_case()
    print("示例输入 x:\n", x)
    print("目标输出（近似）:\n", target)
    print()

    run_all_tests()
