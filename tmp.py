
from copy import deepcopy

def resheto_eratosfena(n: int=10) -> list:
    """
    Args:
        n: в каком диапазоне искать простые числа [2, n]
    Return:
        список простых чисел в дипазоне [2, n]
    """
    # В этот список будем складывать простые числа
    prime_digits = []
    # Создаём список, элементы которого будем "просеивать"
    try_list = list(range(2, n+1))
    # Пока есть что просеивать
    while len(try_list) > 0:
        # Первый элемент этого списка - простое число. Положим его в наш список и удалим из try_list
        prime_elem = try_list.pop(0)
        prime_digits.append(prime_elem)
        # Пробегаем по элементам списка
        for elem in try_list:
            # Если элемент кратен последнему найденному простому числу - удаляем его, elem не простое число
            if elem % prime_elem == 0:
                try_list.remove(elem)
    
    return prime_digits
