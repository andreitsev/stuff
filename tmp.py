import pickle
import os
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm


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



def pickle_object(object_to_save: object, path: str):
    
    """
    Создаёт вложенные папки (если они не существовали до этого) и записывает (pickle.dump) файл
    
    Args:
        object_to_save: что пиклить pickle.dump(object_to_save, ...)
        path: Путь, куда сохранить файл /home/.../<object_to_save>.pkl
    """
    
    previous_path = '/'
    current_path = previous_path
    path_len = len(path.split('/'))

    for i, path_part in enumerate(path.split('/')):
        if path_part == '':
            continue
        
        current_path = os.path.join(current_path, path_part)
        # Если такой папки нет, то
        if path_part not in os.listdir(previous_path):
            # Если это название файла, то запишем его
            if i == path_len - 1:
                pickle.dump(object_to_save, open(current_path, mode='wb'))
            # создать её
            else:
                os.mkdir(current_path)
        if i == path_len - 1:
            pickle.dump(object_to_save, open(current_path, mode='wb'))

        previous_path = current_path
    

def analyse_dataframe(
        df: pd.DataFrame,
        show_unq_vals_threshold: int = 20,
        verbose: bool = True,
        key_val_gap: int = 50,
        columns: list = None
) -> None:

    """
    Показывает метоинформацию по датафрейму df - число строк, число нанов по каждой колонке, уникальные значения
    в каждой колонке

    Args:
        df: pd.DataFrame, для которого сделать диагностику
        show_unq_vals_threshold: сколько уникальных значений переменной показывать
        verbose: визуализировать ли цикл по колонкам
        key_val_gap: ширина '-' между уникальным значением в колонке и её числом встречаний
    """

    print(f'Размер датафрейма: {df.shape[0]:,} x {df.shape[1]:,}')
    cols = df.columns
    if columns is not None:
        cols = columns
    for col in tqdm(cols) if verbose else cols:
        print('=' * 100)
        print(col, end='\n' * 2)
        # Если в колоке дата - сортируем по убыванию даты
        is_datetime_type = str(df[col].dtype).find('datetime') != -1
        unq_vals_dict = df[col].value_counts().sort_index(ascending=False).to_dict() if is_datetime_type else df[
            col].value_counts().to_dict()
        unq_vals_percents_dict = df[col].value_counts(1).sort_index(ascending=False).to_dict() if is_datetime_type else \
        df[col].value_counts(1).to_dict()
        print(' ' * 2, f'Число уникальных значений: {len(unq_vals_dict):,}')
        nans = df[col].isnull().sum()
        print(' ' * 2, f"Число NaN's: {nans:,} ({round(nans * 100 / len(df), 1)}%)")
        print(' ' * 2, 'Уникальные значения:', end='\n' * 2)
        n_ = 0
        for (key1, val1), (key2, val2) in zip(unq_vals_dict.items(), unq_vals_percents_dict.items()):
            print(' ' * 6, key1, '-' * (key_val_gap - len(f"{val1:,} ({round(val2 * 100, 2)}%)") - len(str(key1))),
                  f"{val1:,} ({round(val2 * 100, 2)}%)")
            n_ += 1
            if n_ == show_unq_vals_threshold:
                break
    return





