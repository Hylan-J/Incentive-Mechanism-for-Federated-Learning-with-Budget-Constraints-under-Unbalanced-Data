from .utils import *


def FLIM(trade_information):
    R = trade_information["R"]
    c = trade_information["c"]
    N = len(c)
    X = [0] * N
    P = [0.0] * N

    # 获取排序后的报价
    c_sorted_value, c_sorted_index = sort_with_value(c)

    # 1<=class_num<=num_clients
    for j in range(1, N + 1):
        _index_ = j - 1
        # 如果第j个客户端（下标j-1）的报价大于分给j个客户端的平均预算
        if c_sorted_value[_index_] > (R / j):
            # 1<=index<=class_num-1
            for i in range(1, j):
                __index__ = i - 1
                # 第i个客户端（下标i-1）
                P[__index__] = min(R / (j - 1), c_sorted_value[_index_])
                X[__index__] = 1
            break
        else:
            P[_index_] = c_sorted_value[_index_]
            X[_index_] = 1

    # 返回原报价对应X
    X = restore_with_index(X, c_sorted_index)
    # 返回原报价对应的P
    P = restore_with_index(P, c_sorted_index)

    return X, P
