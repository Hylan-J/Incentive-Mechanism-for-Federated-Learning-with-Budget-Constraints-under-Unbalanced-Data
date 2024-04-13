from .utils import *


def FLIM(information):
    # 当前轮的服务器预算
    current_R = information["R"]
    # 客户端报价
    current_c = information["c"]

    N = len(current_c)
    X, P = [0] * N, [0.0] * N

    # 获取排序后的报价
    c_sorted_value, c_sorted_index = sort_with_value(current_c)

    for i in range(0, N):
        # 如果第i个客户端的报价低于平均分配到的钱，那么这边一定是可以选的
        if c_sorted_value[i] < (current_R / (i + 1)):
            for j in range(0, i + 1):
                if i < N - 1:
                    P[j] = min(current_R / (i + 1), c_sorted_value[i + 1])
                else:
                    P[j] = min(current_R / (i + 1), c_sorted_value[i])
                X[j] = 1

    # 返回原报价对应X
    X = restore_with_index(X, c_sorted_index)
    # 返回原报价对应的P
    P = restore_with_index(P, c_sorted_index)

    return X, P
