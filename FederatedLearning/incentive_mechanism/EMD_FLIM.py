from math import e
from .utils import *


def EMD_FLIM(information):
    # 服务器预算
    current_R = information["R"]
    # 客户端报价
    current_c = information["c"]
    # 客户端数据质量
    EMDs = information["EMDs"]

    N = len(current_c)
    X, P = [0] * N, [0.0] * N

    # 论文中的delta（模型精度相关信息）
    delta = []
    # 论文中的 c/delta
    sigma = []

    for i in range(N):
        delta_each = 1 / e ** EMDs[i]
        delta.append(delta_each)
        sigma.append(current_c[i] / delta_each)

    # 根据sigma从小到大排序
    sigma_sorted_value, sigma_sorted_index = sort_with_value(sigma)
    # 对应的delta根据其排序
    delta_sorted = restore_with_index(delta, sigma_sorted_index)

    for i in range(0, N):
        # 如果第i个客户端的报价低于平均分配到的钱，那么这边一定是可以选的
        if sigma_sorted_value[i] < (current_R / sum(delta_sorted[:i + 1])):
            for j in range(0, i + 1):
                if i < N - 1:
                    P[j] = delta_sorted[j] * min(current_R / sum(delta_sorted[:i + 1]),
                                                 sigma_sorted_value[i+1])
                else:
                    P[j] = delta_sorted[j] * min(current_R / sum(delta_sorted[:i + 1]),
                                                 sigma_sorted_value[i])
                X[j] = 1

    # 返回被挑选结果，对应X
    X = restore_with_index(X, sigma_sorted_index)
    # 返回原报价，对应P
    P = restore_with_index(P, sigma_sorted_index)

    return X, P
