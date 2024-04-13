from math import e
from .utils import *


def FMore(information):
    # 服务器预算
    current_R = information["R"]
    # 客户端报价
    current_c = information["c"]
    # 客户端数据质量
    EMDs = information["EMDs"]

    N = len(current_c)
    X, P = [0] * N, [0.0] * N

    p = current_c  # 预期报酬
    q = []  # 资源向量
    for i in range(N):
        q.append(e ** EMDs[i])
    score = [max_min_normalize(q)[i] - max_min_normalize(q)[i] for i in range(N)]
    # 将得分进行排序
    score_sorted_value, score_sorted_index = sort_with_value(score)
    p_sorted_value = restore_with_index(p, score_sorted_index)

    for i in range(0, N):
        if sum(p_sorted_value[:i+1]) < current_R:
            P[i] = p_sorted_value[i]
            X[i] = 1

    # 返回被挑选结果，对应X
    X = restore_with_index(X, score_sorted_index)
    P = restore_with_index(P, score_sorted_index)

    return X, P
