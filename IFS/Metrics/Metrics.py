import numpy as np


class Metrics:
    def __init__(self, y_true: np.ndarray  # shape (n,)
    ,y_pred: np.ndarray  # shape (n,) —— 二分类预测（0/1）
    ,y_score: np.ndarray  # shape (n,)—— 预测得分
    ):
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_score = y_score
        assert len(y_true) == len(y_pred), "Lengths of y_true and y_pred must be equal"
        assert len(y_true) == len(y_score), "Lengths of y_true and y_score must be equal"
        self.length = len(y_pred)


    def accuracy(self):
        return sum(self.y_pred==self.y_true)/self.length
    
    def precision(self):
        TP = sum((self.y_true == 1) & (self.y_pred==1))
        FP = sum((self.y_true == 0) & (self.y_pred==1))
        return TP/(TP+FP) if (TP+FP)>0 else 0
    
    def recall(self):
        TP = sum((self.y_true == 1) & (self.y_pred==1))
        FN = sum((self.y_true == 1) & (self.y_pred==0))
        return TP/(TP+FN) if (TP+FN)> 0 else 0
    
    def f1(self):
        recall = self.recall()
        precision = self.precision()
        if recall+precision == 0:
            return 0
        return 2*precision*recall/(precision+recall)
    
    def AUC(self):
        """
        AUC是ROC曲线和x周围成的面积
        ROC曲线是TPR和FPR在各个阈值值下取值对形成的连线
        AUC代表着任意取出一个样本对，将正样本打分高于负样本的概率，也反映了将正样本排在负样本前的能力
        所以本质上AUC代表着模型的排序能力
        AUC = sum(每一个阈值下正样本在负样本前的个数）/(正负样本组合数量)
        """
        samples = list(zip(self.y_score, self.y_true))

        samples.sort(key = lambda x: (x[0], -x[1]))

        P = np.sum(self.y_true == 1)
        N = len(samples) - P

        if P == 0 or N == 0:
            return 0.5

        sum_r = 0
        for i, (score, label) in enumerate(samples, start = 1):
            if label == 1:
                sum_r+=i
        return (sum_r - P*(P+1)/2)/(P*N)


def assert_close(actual, expected, name, tol=1e-9):
    if abs(actual - expected) > tol:
        raise AssertionError(f"{name} 错误: 期望 {expected}, 实际 {actual}")


def run_test_case(name, func):
    try:
        func()
        print(f"{name}: 正确")
    except AssertionError as exc:
        print(f"{name}: 错误 -> {exc}")
        raise


def test_basic_metrics():
    y_true = np.array([1, 0, 1, 0, 1, 0])
    y_pred = np.array([1, 1, 1, 0, 0, 0])
    y_score = np.array([0.95, 0.80, 0.70, 0.40, 0.30, 0.10])

    metrics = Metrics(y_true, y_pred, y_score)

    assert_close(metrics.accuracy(), 4 / 6, "accuracy")
    assert_close(metrics.precision(), 2 / 3, "precision")
    assert_close(metrics.recall(), 2 / 3, "recall")
    assert_close(metrics.f1(), 2 / 3, "f1")
    assert_close(metrics.AUC(), 2 / 3, "AUC")


def test_zero_positive_predictions():
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([0, 0, 0, 0])
    y_score = np.array([0.40, 0.30, 0.20, 0.10])

    metrics = Metrics(y_true, y_pred, y_score)

    assert metrics.precision() == 0, "precision 应为 0"
    assert metrics.recall() == 0, "recall 应为 0"
    assert metrics.f1() == 0, "f1 应为 0"


def test_auc_with_single_class():
    cases = [
        (np.array([1, 1, 1]), np.array([1, 1, 1]), np.array([0.9, 0.8, 0.7])),
        (np.array([0, 0, 0]), np.array([0, 0, 0]), np.array([0.3, 0.2, 0.1])),
    ]

    for idx, (y_true, y_pred, y_score) in enumerate(cases, start=1):
        metrics = Metrics(y_true, y_pred, y_score)
        assert_close(metrics.AUC(), 0.5, f"single_class_case_{idx}_AUC")


def test_input_length_validation():
    try:
        Metrics(
            np.array([1, 0, 1]),
            np.array([1, 0]),
            np.array([0.9, 0.8, 0.7]),
        )
    except AssertionError:
        pass
    else:
        raise AssertionError("y_true 和 y_pred 长度不一致时应抛出 AssertionError")

    try:
        Metrics(
            np.array([1, 0, 1]),
            np.array([1, 0, 1]),
            np.array([0.9, 0.8]),
        )
    except AssertionError:
        pass
    else:
        raise AssertionError("y_true 和 y_score 长度不一致时应抛出 AssertionError")


if __name__ == "__main__":
    test_cases = [
        ("基础指标测试", test_basic_metrics),
        ("无正类预测测试", test_zero_positive_predictions),
        ("单一类别 AUC 测试", test_auc_with_single_class),
        ("输入长度校验测试", test_input_length_validation),
    ]

    for case_name, case_func in test_cases:
        run_test_case(case_name, case_func)

    print("全部测试通过")
