import numpy as np
from ActivationFunction import softmax
import torch 
import torch.nn.functional as F

def kl_divergence(p , q, eps=1e-10):
    """
    计算 KL 散度 D_KL(p || q)

    参数：
        p: numpy array，真实分布
        q: numpy array，近似分布
        eps: 防止 log(0) 的小值

    返回：
        KL 散度（标量）
    """
    p = np.clip(p, eps, 1)
    q = np.clip(q, eps, 1)

    return np.sum(p * np.log(p/ q))

def cross_entropy_loss(logits, labels):
    """
    :param logits: 原始输出 (N, C)
    :param labels: 标签，（N, ）
    :return: 批量损失
    """
    probs = softmax(logits)
    batch_size = logits.shape[0]
    true_probs = probs[np.arange(batch_size), labels]
    return -np.sum(np.log(true_probs+1e-8))/batch_size


def InfoNCE(queries: torch.Tensor# (B, D) 第 i 个 query 的正样本是第 i 个 key
            , keys: torch.Tensor# (B, D)
            , temperature:float):
    queries = F.normalize(queries, dim = -1)
    keys = F.normalize(keys, dim = -1)
    sim = queries @ keys.T / temperature

    labels = torch.arange(queries.shape[0])
    return F.cross_entropy(sim, labels)


if __name__ == "__main__":
    # #定义两个简单的概率分布         
    # p = np.array([0.2, 0.5, 0.3]) 
    # q = np.array([0.1, 0.7, 0.2]) # 近似分布
    
    # #计算 KL 散度
    # kl_pq = kl_divergence(p, q)
    # kl_qp = kl_divergence(q, p)
    
    # print(f"KL(p||q) = {kl_pq:.4f}") # 输出: KL(p||q) = 0.0619
    # print(f"KL(q||p) = {kl_qp:.4f}") # 输出: KL(qlp) = 0.0657

    # 模型原始输出 (2个样本, 3个类别)
    logits = np.array([
        [1.2, 0.8, 0.1],
        [0.3, 2.1, 1.5]
    ])

    # 真实标签（非 one-hot）
    # 第1个样本类别0，第2个样本类别1
    labels = np.array([0, 1])

    # 计算损失
    loss = cross_entropy_loss(logits, labels)

    print(f"多分类交叉熵损失值: {loss:.4f}")  # 约 0.4152