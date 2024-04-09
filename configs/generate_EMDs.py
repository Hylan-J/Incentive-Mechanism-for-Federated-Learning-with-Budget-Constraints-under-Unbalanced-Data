import random
import numpy as np
import matplotlib.pyplot as plt


def generate_EMDs(divide_type):
    divide_parameter = {
        "a": {"mean": 0.52, "0-0.5": 10, "0.5-1": 10},
        "b": {"mean": 0.40, "0-0.5": 15, "0.5-1": 5},
        "c": {"mean": 0.27, "0-0.5": 20, "0.5-1": 0}
    }

    mean = divide_parameter[divide_type]["mean"]
    low = divide_parameter[divide_type]["0-0.5"]
    high = divide_parameter[divide_type]["0.5-1"]

    EMDs = []
    generate_done = False
    while not generate_done:
        if low != 0:
            for i in range(low):
                EMDs.append(random.uniform(0, 0.5))
        if high != 0:
            for i in range(high):
                EMDs.append(random.uniform(0.5, 1))
        if abs(sum(EMDs) / (low + high) - mean) < 0.001:
            generate_done = True
        else:
            EMDs = []
    return EMDs


"""
import numpy as np
from scipy.optimize import minimize


def generate_EMDs(mean, low, high):
    # 定义目标函数，这里将函数的平均值与目标值mean的差的平方进行最小化
    def objective(x):
        return np.abs((np.mean(x) - mean))

    # 定义约束条件：a个小于0.5，b个大于等于0.5
    def constraint1(x):
        return low - np.sum(x < 0.5)

    def constraint2(x):
        return high - np.sum(x >= 0.5)

    # 初始猜测为随机生成的长度为a+b的列表
    x0 = np.random.rand(low + high)

    cons = [{'type': 'eq', 'fun': constraint1},{'type': 'eq', 'fun': constraint2}]
    sol = minimize(objective, x0, method='SLSQP', constraints=cons)

    return sol.x.tolist()

EMDs = generate_EMDs(0.40,15,5)
print(EMDs)
print(np.sum(np.array(EMDs) < 0.5))
print(np.mean(EMDs))
"""

if __name__ == "__main__":
    Options_Divide = ["a", "b", "c"]
    EMDs = generate_EMDs(Options_Divide[0])
    # 打乱顺序
    random.shuffle(EMDs)
    print("生成的EMDs:\n", EMDs)
    print("EMDs均值:\n", np.mean(EMDs))
    np.save(f"EMDs_d.npy", EMDs)

    plt.figure()
    plt.plot(EMDs, "b-*")
    plt.show()
