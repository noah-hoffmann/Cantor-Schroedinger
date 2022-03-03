from math import sqrt


class SymmetricIndex:
    def __init__(self, limit):
        self.limit = limit
        self.length = limit * (limit + 1) // 2

    def __len__(self):
        return self.length

    @staticmethod
    def get_index(*indices):
        i, j = indices
        return j + i * (i + 1) // 2

    @staticmethod
    def get_indices(index: int):
        i = int((-1 + sqrt(1 + 8 * index)) / 2)
        j = index - i * (i + 1) // 2
        return i, j

    def __iter__(self):
        for i in range(self.length):
            yield i, self.get_indices(i)

    def __call__(self, *args):
        if len(args) == 1:
            return self.get_indices(args[0])
        elif len(args) == 2:
            return self.get_index(*args)


def main():
    converter = SymmetricIndex(4)
    for i in range(4):
        for j in range(i + 1):
            I = converter.get_index(i, j)
            print(f'({i}, {j}) -> {I} -> {converter.get_indices(I)}')


if __name__ == '__main__':
    main()
