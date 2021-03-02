import numpy as np

def permute_2ndarray_by_1starray(arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray:
    """
    Args:
        arr1 - список (np.array) из уникальных элементов
        arr2 - список (np.array) из тех же, елементов, что и arr1, но в другом порядке
        (Иными словами: len(set(arr1) & set(arr2)) == len(arr1) == len(arr2))
    Return:
        resulting_permutation - список (np.array), содержащий перестановку, такую что:
        arr1 == arr2[resulting_permutation]
    """
    assert len(set(arr1)) == len(arr1), 'В arr1 не все элементы уникальны!'
    assert len(set(arr2)) == len(arr2), 'В arr2 не все элементы уникальны!'
    assert len(set(arr1) & set(arr2)) == len(arr1), 'Множество элементов в arr1 и arr2 не совпадают!'

    resulting_permutation = []
    for elem in arr1:
        current_position = np.where(arr2 == elem)[0][0]
        resulting_permutation.append(current_position)
    return np.array(resulting_permutation)

"""
Пример использования:

a = np.array([2, 1, 4, 3])
b = np.array([3, 2, 1, 4])
res = permute_2ndarray_by_1starray(arr1=a, arr2=b)
print(res)

    [1, 2, 3, 0]
    
b[res] = a
"""
