import os

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
