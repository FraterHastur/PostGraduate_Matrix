import numpy as np


class LU_decay:
    def __init__(self, input_matrix_a, input_matrix_b):
        self.input_matrix_a = input_matrix_a.astype(np.float)
        self.input_matrix_b = input_matrix_b.astype(np.float).T
        self.l_list = []

    def __try_matrix_dimension(self, input_matrix=None):
        if input_matrix is None:
            input_matrix = self.input_matrix_a

        m, n = input_matrix.shape

        if m != n:
            return False
        elif len(input_matrix.shape) != 2:
            return False
        else:
            return True

    def __try_minor_not_null(self, input_matrix=None):
        if input_matrix is None:
            input_matrix = self.input_matrix_a

        if self.__try_matrix_dimension(input_matrix) is not False:  # todo сделать отдельный обработчик ошибок
            new_matrix = np.delete(input_matrix, 0, axis=0)
            new_matrix = np.delete(new_matrix, 0, axis=1)

            det = np.linalg.det(new_matrix)

            if det != 0:
                return True
            else:
                return False
        else:
            return 'Матрица не квадратная'

    def make_u_matrix(self, input_matrix=None):
        if input_matrix is None:
            input_matrix = self.input_matrix_a
        else:
            input_matrix = input_matrix.astype(np.float)
        # todo сделать проверку условий для работы метода Гаусса
        cnt = 0
        for row in range(0, len(input_matrix[0])):
            self.l_list.append(input_matrix[row][row])
            input_matrix[row] = input_matrix[row] / input_matrix[row][row]
            cnt += 1
            for val in range(cnt, len(input_matrix[0])):
                self.l_list.append(input_matrix[val][row])
                input_matrix[val] = input_matrix[val] - (input_matrix[row] * input_matrix[val][row])
        return input_matrix

    def make_l_matrix(self, input_matrix=None):
        if input_matrix is None:
            input_matrix = self.input_matrix_a
        else:
            input_matrix = input_matrix.astype(np.float)

        if len(self.l_list) == 0:
            self.make_u_matrix(input_matrix)
            cnt = 0
            shape, _ = input_matrix.shape
            output_matrix = np.zeros((shape, shape))
            for i in range(0, shape):
                for j in range(cnt, shape):
                    output_matrix[j][i] = self.l_list.pop(0)
                cnt += 1
            return output_matrix
        else:
            cnt = 0
            shape, _ = input_matrix.shape
            output_matrix = np.zeros((shape, shape))
            for i in range(0, shape):
                for j in range(cnt, shape):
                    output_matrix[j][i] = self.l_list.pop(0)
                cnt += 1
            return output_matrix

    def make_ly(self, input_matrix_a=None, input_matrix_b=None):
        if input_matrix_a is None:
            input_matrix_a = self.input_matrix_a
        else:
            input_matrix_a = input_matrix_a.astype(np.float)

        if input_matrix_b is None:
            input_matrix_b = self.input_matrix_b
        else:
            input_matrix_b = input_matrix_b.astype(np.float)

        return np.linalg.solve(input_matrix_a, input_matrix_b)

    def make_ux(self, input_matrix_a=None, input_matrix_b=None):
        if input_matrix_a is None:
            input_matrix_a = self.input_matrix_a
        else:
            input_matrix_a = input_matrix_a.astype(np.float)

        if input_matrix_b is None:
            input_matrix_b = self.make_ly(input_matrix_a, self.input_matrix_b)
        else:
            input_matrix_b = input_matrix_b.astype(np.float)
        x_sol = np.linalg.solve(input_matrix_a, input_matrix_b)
        return x_sol

    def make_lu_solve(self, input_matrix_a=None, input_matrix_b=None):
        if input_matrix_a is None:
            input_matrix_a = self.input_matrix_a
        else:
            input_matrix_a = input_matrix_a.astype(np.float)

        if input_matrix_b is None:
            input_matrix_b = self.input_matrix_b
        else:
            input_matrix_b = input_matrix_b.astype(np.float)

        if self.__try_matrix_dimension(input_matrix_a) is True and self.__try_minor_not_null(input_matrix_a) is True:
            u_matrix = self.make_u_matrix(input_matrix_a)
            l_matrix = self.make_l_matrix(input_matrix_a)
            l_solve = self.make_ly(l_matrix, input_matrix_b)
            print(f'Ly= {l_solve}')
            u_solve = self.make_ux(u_matrix, l_solve)
            return u_solve


if __name__ == '__main__':
    matrix = np.array([[2, 1, 3], [11, 7, 5], [9, 8, 4]])
    matrix2 = np.array([[1, -6, -5]])
    test = LU_decay(matrix, matrix2)
    print(test.make_l_matrix())
    print(test.make_u_matrix())
    print(test.make_lu_solve())
