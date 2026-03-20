import torch
from torch import nn


class BatchNorm2d(nn.Module):
    """
    一个最小版 BatchNorm2d。

    输入形状：
        x: (N, C, H, W)
            N = batch size
            C = channel 数
            H = height
            W = width

    参数含义：
        num_features: 通道数 C
        eps: 防止除零
        momentum: running statistics 的动量

    说明：
        BatchNorm2d 是“按通道”做归一化。
        对每个通道 c，统计该通道在 (N, H, W) 这三个维度上的均值和方差。

        训练时：
            mean = 当前 batch 的均值，shape = (1, C, 1, 1)
            var  = 当前 batch 的方差，shape = (1, C, 1, 1)
            x_hat = (x - mean) / sqrt(var + eps)
            y = gamma * x_hat + beta

            并更新：
            running_mean = momentum * running_mean + (1 - momentum) * mean
            running_var  = momentum * running_var  + (1 - momentum) * var

        推理时：
            使用 running_mean / running_var 做归一化。
    """

    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        # 可学习参数：每个通道一个缩放 gamma、一个平移 beta
        # 使用 (1, C, 1, 1) 方便与 (N, C, H, W) 广播
        self.gamma = nn.Parameter(torch.ones((1, num_features, 1, 1)))
        self.beta = nn.Parameter(torch.zeros((1, num_features, 1, 1)))

        # running statistics，推理阶段使用
        self.register_buffer('running_mean', torch.zeros((1, num_features, 1, 1)))
        self.register_buffer('running_var', torch.ones((1, num_features, 1, 1)))

    def forward(self, x, training=True):
        """
        参数:
            x: shape (N, C, H, W)
            training: True 表示训练模式，False 表示推理模式

        返回:
            y: shape (N, C, H, W)
        """
        assert x.ndim == 4, f"expect 4D input, got shape={x.shape}"
        assert x.shape[1] == self.num_features, (
            f"expect channel dim = {self.num_features}, got {x.shape[1]}"
        )

        if training:
            # =========================
            # TODO 1: 计算当前 batch 的均值 mean
            # 要求：对每个通道，沿 (N, H, W) 做统计
            # 并保持形状为 (1, C, 1, 1)
            # =========================
            mean = x.mean(dim = (0, 2, 3), keepdim = True)
            # =========================
            # TODO 2: 计算当前 batch 的方差 var
            # 要求：对每个通道，沿 (N, H, W) 做统计
            # 并保持形状为 (1, C, 1, 1)
            # 这里使用“总体方差”即可，也就是 unbiased=False
            # =========================
            var = x.var(dim = (0,2,3), keepdim = True, unbiased = False)
            # =========================
            # TODO 3: 根据 mean / var 对 x 做标准化
            # x_hat = (x - mean) / sqrt(var + eps)
            # =========================
            x_hat = (x - mean)/torch.sqrt(var+self.eps)
            # =========================
            # TODO 4: 仿射变换输出
            # y = gamma * x_hat + beta
            # =========================
            y = self.gamma*x_hat+self.beta
            # =========================
            # TODO 5: 更新 running_mean
            # running_mean = momentum * running_mean + (1 - momentum) * mean
            # =========================
            self.running_mean = self.momentum*self.running_mean + (1- self.momentum)*mean
            # =========================
            # TODO 6: 更新 running_var
            # running_var = momentum * running_var + (1 - momentum) * var
            # =========================
            self.running_var = self.momentum*self.running_var + (1 - self.momentum)*var

            return y

        else:
            # =========================
            # TODO 7: 推理阶段使用 running_mean / running_var 标准化
            # x_hat = (x - running_mean) / sqrt(running_var + eps)
            # =========================
            x_hat = (x - self.running_mean)/torch.sqrt(self.running_var+self.eps)
            # =========================
            # TODO 8: 推理阶段输出仿射变换结果
            # y = gamma * x_hat + beta
            # =========================
            y = self.gamma*x_hat+self.beta
            return y


# =========================
# 参考输入 / 目标输出（方便你手算验证）
# =========================
def demo_case():
    """
    一个非常小的例子，便于你理解每一步。

    输入形状：
        x.shape = (1, 2, 2, 1)

    第 1 个通道的数据是:
        [[1],
         [3]]
    均值 = 2, 方差 = 1

    第 2 个通道的数据是:
        [[2],
         [4]]
    均值 = 3, 方差 = 1

    标准化后约为：
        第 1 个通道:
            [[-1],
             [ 1]]
        第 2 个通道:
            [[-1],
             [ 1]]

    因为初始 gamma=1, beta=0，所以输出 y ≈ x_hat
    """
    x = torch.tensor(
        [
            [
                [[1.0], [3.0]],
                [[2.0], [4.0]],
            ]
        ]
    )

    target = torch.tensor(
        [
            [
                [[-1.0], [1.0]],
                [[-1.0], [1.0]],
            ]
        ]
    )

    return x, target


# =========================
# 测试代码
# 你补完 TODO 后，直接运行本文件即可
# =========================
def test_shape():
    bn = BatchNorm2d(num_features=3)
    x = torch.randn(8, 3, 5, 5)
    y = bn.forward(x, training=True)
    assert y.shape == x.shape


def test_training_normalization():
    """
    当 gamma=1, beta=0 时，训练态输出应接近“每个通道均值为 0、方差为 1”。
    """
    bn = BatchNorm2d(num_features=4)
    x = torch.randn(16, 4, 6, 6)
    y = bn.forward(x, training=True)

    channel_mean = y.mean(dim=(0, 2, 3))
    channel_var = y.var(dim=(0, 2, 3), unbiased=False)

    assert torch.allclose(channel_mean, torch.zeros(4), atol=1e-6), channel_mean
    assert torch.allclose(channel_var, torch.ones(4), atol=1e-4), channel_var


def test_demo_case_close():
    bn = BatchNorm2d(num_features=2, eps=1e-5)
    x, target = demo_case()
    y = bn.forward(x, training=True)

    # 由于 eps 存在，数值不会严格等于 ±1，所以这里放宽一点
    assert torch.allclose(y, target, atol=1e-3), y


def test_running_stats_updated():
    bn = BatchNorm2d(num_features=2, momentum=0.1)
    x, _ = demo_case()
    _ = bn.forward(x, training=True)

    # 当前 batch:
    # mean = [[[[2]], [[3]]]]
    # var  = [[[[1]], [[1]]]]
    # 初始 running_mean = 0, running_var = 1
    # 更新后:
    # running_mean = 0.1 * 0 + 0.9 * mean = [[[[1.8]], [[2.7]]]]
    # running_var  = 0.1 * 1 + 0.9 * 1 = [[[[1]], [[1]]]]
    expected_running_mean = torch.tensor([[[[1.8]], [[2.7]]]])
    expected_running_var = torch.tensor([[[[1.0]], [[1.0]]]])

    assert torch.allclose(bn.running_mean, expected_running_mean, atol=1e-6), bn.running_mean
    assert torch.allclose(bn.running_var, expected_running_var, atol=1e-6), bn.running_var


def test_inference_uses_running_stats():
    bn = BatchNorm2d(num_features=2, momentum=0.1)

    # 先喂一个 batch，让 running stats 更新
    x_train, _ = demo_case()
    _ = bn.forward(x_train, training=True)

    # 推理时应使用 running_mean=[1.8, 2.7], running_var=[1,1]
    x_test = torch.tensor(
        [
            [
                [[1.8], [2.8]],
                [[2.7], [3.7]],
            ]
        ]
    )
    y = bn.forward(x_test, training=False)

    # 每个通道里：
    # 第一格正好等于 running mean，归一化后应接近 0
    # 第二格比 running mean 大 1，归一化后应接近 1
    expected = torch.tensor(
        [
            [
                [[0.0], [1.0]],
                [[0.0], [1.0]],
            ]
        ]
    )
    assert torch.allclose(y, expected, atol=1e-3), y


def run_all_tests():
    test_shape()
    test_training_normalization()
    test_demo_case_close()
    test_running_stats_updated()
    test_inference_uses_running_stats()
    print("All tests passed! 你已经完成了 BatchNorm2d 的基础实现。")


if __name__ == "__main__":
    print("===== BatchNorm2d 填空练习 =====")
    x, target = demo_case()
    print("示例输入 x:\n", x)
    print("目标输出（近似）:\n", target)
    print()

    run_all_tests()
