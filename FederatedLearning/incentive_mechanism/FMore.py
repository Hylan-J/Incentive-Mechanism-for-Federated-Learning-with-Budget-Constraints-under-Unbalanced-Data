from math import e
from .utils import *


def FMore(server, clients):
    # 获取客户端个数
    num = len(clients)
    # 分配数组X
    X = [0] * num
    # 支付数组P
    P = [0.0] * num

    p = []  # 预期报酬
    q = []  # 资源向量
    for client in clients:
        p.append(client.C)
        q.append(e ** client.EMD)
    score = [max_min_normalize(q)[i] - max_min_normalize(q)[i] for i in range(num)]
    # 将得分进行排序
    score_sorted_value, score_sorted_index = sort_with_value(score)
    p_sorted_value = restore_with_index(p, score_sorted_index)

    for j in range(1, num + 1):
        if sum(p_sorted_value[:j]) < server.R:
            P[j - 1] = p_sorted_value[j - 1]
            X[j - 1] = 1
    # 返回被挑选结果，对应X
    X = restore_with_index(X, score_sorted_index)
    P = restore_with_index(P, score_sorted_index)

    return X, P
