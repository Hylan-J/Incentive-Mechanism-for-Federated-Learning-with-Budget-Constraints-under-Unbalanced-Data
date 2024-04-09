def sort_with_value(a):
    """
    对列表a进行升序排序，返回排序后的列表a和对应的下标关系
    """
    index = list(range(len(a)))
    result = [(a[i], index[i]) for i in range(len(a))]
    result.sort()
    return [r[0] for r in result], [r[1] for r in result]


def restore_with_index(sorted_a, index):
    """
    对排好序后的列表sorted_a，根据下标对应关系还原，返回原数组a
    """
    result = [0] * len(sorted_a)
    for i, idx in enumerate(index):
        result[idx] = sorted_a[i]
    return result


def max_min_normalize(a):
    MAX = max(a)
    MIN = min(a)
    if MAX > MIN:
        b = [(i - MIN) / (MAX - MIN) for i in a]
    else:
        b = [1] * len(a)
    return b