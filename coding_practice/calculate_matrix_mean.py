def calculate_matrix_mean(matrix: list[list[float]], mode: str) -> list[float]:
    if mode == 'column':
        means = [None for c in range(len(matrix[0]))]
        for c_idx in range(len(matrix[0])):
            sum = 0
            for r_idx in range(len(matrix)):
                sum += matrix[r_idx][c_idx]
            means[c_idx] = sum /  len(matrix)         
    else:
        means = [None for r in range(len(matrix))]
        for r_idx in range(len(matrix)):
            sum = 0
            for c_idx in range(len(matrix[0])):
                sum += matrix[r_idx][c_idx]
            means[r_idx] = sum /  len(matrix[0]) 
    return means