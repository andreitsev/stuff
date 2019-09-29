import numpy as np
from copy import deepcopy
from math import gcd

# Находит НОД для списка из чисел
def multiple_gcd(elements_list):
    if len(elements_list) == 1:
        common_gcd = [elements_list]
    else:
        common_gcd = [gcd(abs(elements_list[0]), abs(elements_list[1]))]
        for i in range(2, len(elements_list)):
            common_gcd.append(gcd(common_gcd[-1], abs(elements_list[i])))
    return common_gcd[-1]

def lcm(a, b):
    return int(abs(a*b)/gcd(a, b))

def multiple_lcm(elements_list):
    if len(elements_list) == 1:
        common_lcm = [elements_list]
    else:
        common_lcm = [lcm(abs(elements_list[0]), abs(elements_list[1]))]
        for i in range(2, len(elements_list)):
            common_lcm.append(lcm(common_lcm[-1], abs(elements_list[i])))
    return common_lcm[-1]

class GaussElimination(object):
    
    def __init__(self, A, b=None):
        self.A = A
        self.b = b
        self.reduction_matrix = None
        self.basis_element_index = []
        self.basis_matrix = None
        self.used_rows = []
        self.used_cols = []
        
    def make_system_from_list(self, linear_eq_system):
        """
        Пример: linear_eq_system = ["2*x1 - 3*x2 + 5*x3 = 3", 
                                    "-1*x1 + 1*x3 = 0", 
                                    "11*x2 - 4*x5 = -2"]
        """
        
        # Находим переменную с самым большим индексом
        n_variables = 0
        for eq in linear_eq_system:
            for var in re.findall(r'x\d+', eq):
                n_variables = max(n_variables, int(var.replace('x', '')))

        A = np.zeros((len(linear_eq_system), n_variables))
        b = np.zeros(len(linear_eq_system))
        for i, eq in enumerate(linear_eq_system):
            previous_elem = ''
            for elem in eq.split():
                if 'x' in elem:
                    A[i, int(elem.split('*')[-1][1:])-1] = int(previous_elem + elem.split('*')[0])
                    previous_elem = elem
                elif elem in ['+', '-', '=']:
                    previous_elem = elem
                elif previous_elem == '=':
                    b[i] = int(elem)
        return A, b

        

    def gauss_iteration(self, matrix):
        """
        matrix: матрица для которой считается одна итерация Гаусса-Жордана
        Не использует строки, которые уже были в self.used_rows
        """
        # Оставим уникальные и ненулевые строки в матрице:
        reduced_matrix = []
        for row in matrix:
            if (row.tolist() not in reduced_matrix) and (abs(row).sum() != 0):
                reduced_matrix.append(row.tolist())
        reduced_matrix = np.array(reduced_matrix).astype(float)

        unused_rows = [i for i in range(reduced_matrix.shape[0]) if i not in self.used_rows]
        unused_rows = [i for i in unused_rows if abs(reduced_matrix[i]).sum() != 0]
        unused_cols = [i for i in range(reduced_matrix.shape[1]) if i not in self.used_cols]

        # Если нет неиспользованных строк, или размер использованных столбцов совпадает с кол-вом столбцов в матрице - заканчиваем итерацию
        if len(unused_rows) == 0 or len(self.used_cols) == self.A.shape[1]:
            return reduced_matrix, self.used_rows, self.used_cols
        
        basis_element_index = np.where(reduced_matrix[unused_rows[0]] != 0)[0][0]
        basis_element = reduced_matrix[unused_rows[0], basis_element_index]
        # Основная итерация
        for n, row in enumerate(reduced_matrix):
            if (row != reduced_matrix[unused_rows[0]]).sum() != 0:
                current_element = row[basis_element_index]
                reduced_matrix[n] = basis_element*row - current_element*\
                                    reduced_matrix[unused_rows[0]]

        self.used_rows.append(unused_rows[0])
        self.used_cols.append(basis_element_index)
        return reduced_matrix, self.used_rows, self.used_cols

    def gauss_reduction(self, track_iterations=False):
        """
        метод для получения reduced row echelon form для матрицы A -> self.reduction_matrix
        """
        self.reduction_matrix = deepcopy(self.A)
        basic_elements_indexes = []
        k = 0
        while len(self.used_rows) != len(self.reduction_matrix):
            self.reduction_matrix, self.used_rows, self.used_cols = self.gauss_iteration(self.reduction_matrix)
            k += 1
            if track_iterations:
                print('iteration:', k)
                print('current matrix:')
                print(self.reduction_matrix)
                print('used_rows:', self.used_rows, 'used_cols:', self.used_cols)
                print('_____________________________________')
        # Поделим числа в reduction_matrix на их НОД в каждой строке
        for n, i in enumerate(self.reduction_matrix):
            if (i%1).sum() == 0:
                common_divisor = multiple_gcd([int(j) for j in i if j != 0])
                self.reduction_matrix[n] = i/common_divisor
        
        
        self.basic_elements_indexes = np.array([[i, j] for i, j in zip(self.used_rows, self.used_cols)])
        return self.reduction_matrix, self.basic_elements_indexes

    def linspace_basis(self):
        if self.reduction_matrix is None:
            self.reduction_matrix, self.basic_elements_indexes = self.gauss_reduction()
        
        if self.reduction_matrix.shape[0] >= self.reduction_matrix.shape[1]:
            print('Размерность подпространства равна нулю')
            return 
        else:
            # Отсортируем индексы базисных элементов по столбцу
            sorted_rows = np.argsort(self.basic_elements_indexes[:, 1])
            self.basic_elements_indexes = self.basic_elements_indexes[sorted_rows]
            # Индексы свободных переменных
            independent_variable_columns_indexes = np.array([i for i in range(self.reduction_matrix.shape[1]) if i not in self.basic_elements_indexes[:, 1]])
            # Составление матрицы, где столбцы - базисные вектора подпространства
            self.basis_matrix = np.zeros((self.reduction_matrix.shape[1], 
                                     len(independent_variable_columns_indexes)))
            basic_elements = np.array([self.reduction_matrix[i[0], i[1]] for i in self.basic_elements_indexes]).reshape(-1, 1)

            lcm_ = multiple_lcm(basic_elements.astype(int).ravel())
            self.basis_matrix[self.basic_elements_indexes[:, 1]] = -self.reduction_matrix[sorted_rows][:, independent_variable_columns_indexes]*lcm_/basic_elements
            self.basis_matrix[independent_variable_columns_indexes] = np.eye(len(independent_variable_columns_indexes))*lcm_
            return self.basis_matrix
    
