def matrix_dot_vector(a:list[list[int|float]],b:list[int|float])-> list[int|float]:
    if len(a[0]) != len(b):
        return -1
    c = []
    for r_val in a:
        res = 0
        for idx, r in enumerate(r_val):
            res += r * b[idx]
        c.append(res)
    return c