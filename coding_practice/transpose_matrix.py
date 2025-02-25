def transpose_matrix(a: list[list[int|float]]) -> list[list[int|float]]:
    b = [[None for c in range(len(a))] for r in range(len(a[0]))]
    for r_idx, r in enumerate(a):
        for c_idx, c in enumerate(r):
            if r_idx == c_idx:
                b[r_idx][c_idx] = a[r_idx][c_idx]
            else:
                b[c_idx][r_idx] = a[r_idx][c_idx]
    return b