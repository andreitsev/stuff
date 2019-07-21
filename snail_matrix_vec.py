def matrix2snail_vec(matrix, snail_indexes = []):
    """
    matrix: матрица, которую нужно выпрямить в "улиточный" вектор. 
    return: возвращает вектор snail_indexes - матрица matrix, выпрямленная по правилу ниже
       0 - 1  - 2 -  3
                     |
      11 - 12 - 13 - 4
       |         |   |
      10  15  - 14   5
       |             |
       9 - 8  - 7 -  6
    """
    if matrix.shape[0] >= 3:
        for i in range(matrix.shape[1]):
            snail_indexes.append(matrix[0, i])
        for j in range(1, matrix.shape[0]):
            snail_indexes.append(matrix[j, i])
        for k in range(matrix.shape[1]-1)[::-1]:
            snail_indexes.append(matrix[j, k])
        for d in range(1, matrix.shape[0]-1)[::-1]:
            snail_indexes.append(matrix[d, k])
        snail_indexing(matrix[1:-1, 1:-1], snail_indexes=snail_indexes)
    elif matrix.shape[0] == 1:
        matrix2snail_vec.append(matrix[0, 0])
        
    return np.array(snail_indexes)

# vec = np.random.randint(-4, 4, size=25)
# n = np.sqrt(vec.shape[0])
def snail2matrix(vec, matrix=np.zeros((n, n))):
    """
    vec: вектор из которого заполнить матрицу (длины n^2)
    matrix:  изначально матрица, заполненная нулями. 
    return: возвращает матрицу matrix заполненную из vec по улитке
       0 - 1  - 2 -  3
                     |
      11 - 12 - 13 - 4
       |         |   |
      10  15  - 14   5
       |             |
       9 - 8  - 7 -  6
    """
    new_vec = vec
    new_matrix = matrix
    if new_matrix.shape[0] > 1:
        global_lenght = 0
        for ite, i in enumerate(range(matrix.shape[1])):
            new_matrix[0, i] = new_vec[i]
            global_lenght += 1
        new_vec = new_vec[ite:]
        for ite, j in enumerate(range(1, matrix.shape[0])):
            new_matrix[j, i] = new_vec[ite+1]
            global_lenght += 1
        new_vec = new_vec[ite:]
        for ite, k in enumerate(range(matrix.shape[1]-1)[::-1]):
            new_matrix[j, k] = new_vec[ite+2]
            global_lenght += 1
        new_vec = new_vec[ite:]
        for ite, d in enumerate(range(1, matrix.shape[0]-1)[::-1]):
            new_matrix[d, k] = new_vec[ite+3]
            global_lenght += 1
        vec = vec[global_lenght:]
        snail2matrix(vec, matrix[1:-1, 1:-1])
    elif matrix.shape[0] == 1:
        matrix[0, 0] = new_vec[0]    
    
    return matrix