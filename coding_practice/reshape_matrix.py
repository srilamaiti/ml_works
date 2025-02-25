def reshape_matrix(a: list[list[int|float]], new_shape: tuple[int, int]) -> list[list[int|float]]:
    #Write your code here and return a python list after reshaping by using numpy's tolist() method
    if len(a) * len(a[0]) != new_shape[0] * new_shape[1]:
        return []
    new_r, new_c = new_shape
    reshaped_matrix = [[None for c in range(new_c)] for r in range(new_r)]
    flattened_list = [a[r][c] for r in range(len(a)) for c in range(len(a[0]))]
    r_idx, c_idx = 0, 0
    for i in flattened_list:
        reshaped_matrix[r_idx][c_idx] = i
        if c_idx + 1 == new_c:
            r_idx += 1
            c_idx = 0
        else:
            c_idx += 1
    return reshaped_matrix