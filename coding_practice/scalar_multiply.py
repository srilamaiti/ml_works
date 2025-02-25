def scalar_multiply(matrix: list[list[int|float]], scalar: int|float) -> list[list[int|float]]:
    result = [[None for c in range(len(matrix[0]))] for r in range(len(matrix))]
    for r_idx in range(len(matrix)):
        for c_idx in range(len(matrix[0])):
            result[r_idx][c_idx] = matrix[r_idx][c_idx] * scalar
    return result