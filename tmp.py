
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
