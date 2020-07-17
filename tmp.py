import os
import numpy as np

def prime_decomposition(number: int=10) -> dict:
    """
    Args:
        number - число, которое необходимо разложить на простые множители
    Return:
        словарь, ключами которого являются простые множители числа number, а значения - их кратность
    """
    
    end_range = number + 1
    decomposition = []
    for i in range(2, end_range):
        if number % i == 0:
            while number % i == 0:
                number = number/i
                decomposition.append(i)
                
    decomposition_dict = {}
    for val in decomposition:
        if val not in decomposition_dict:
            decomposition_dict[val] = 1
        else:
            decomposition_dict[val] += 1
    return decomposition_dict


def show_tree_files(path: str, depth: int=0, ignore_folders: tuple=('.parquet',)):
    """
    Args:
        path: путь до места, откуда начинать сканирование документов
        depth: изначальная глубина - (в месте path глубина равна 0)
        ignore_folders: какие метки игнорировать. Если встречается одна из таких меток в директории, 
        то не показывать содержимое этой директории
    Return:
        печатает все файлы с их внутренностями
    """
    
    pre_print = '--'*depth
    for file_name in os.listdir(path):
        print(pre_print, file_name)
        joined_path = os.path.join(path, file_name)
        print_next = sum([ignore_tag in joined_path for ignore_tag in ignore_folders])
        if os.path.isdir(joined_path) and print_next == 0:
              show_tree_files(path=joined_path, depth=depth+1)
    print()

def permute_2ndarray_by_1starray(arr1: np.array, arr2: np.array) -> np.array:
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
    
b[res] -> a
"""


def generate_from_circle(number_of_points: int, center: list, radius: int) -> np.array:
    """
    Генерирует точки из круга: (x - center[0])^2 + (y - center[1])^2 <= radius^2
    
    Args:
        number_of_points - сколько точек генерировать
        center - два числа, центр круга
        radius - число
    Returns:
        np.array размерности (number_of_points x 2)
    """
    
    k = number_of_points
    R = radius
    y = np.linspace(-R,R,10000)
    F = lambda x: (1/np.pi)*np.arcsin(x/R)+(1/np.pi)*(x/R)*np.sqrt(1-(x/R)**2)+1/2
    F_rasp = F(y)
    res = []
    for i in range(k):
        y_samp = y[np.where(F_rasp <= np.random.uniform())[0][-1]]
        x_samp = 2*np.sqrt(R**2 - y_samp**2)*(np.random.uniform()-1/2)
        res.append([x_samp,y_samp])
    res = np.array(res)
    center_ = np.array(center)[np.newaxis,:] * np.ones((k,2))
    return res + center_


