import numpy as np


class MultiIndex:
    def __init__(self, limits: tuple):
        self.limits = limits
        self.length = 1
        for i in limits:
            self.length *= i

    def get_index(self, indices: tuple):
        index = indices[0]
        factor = self.limits[0]          
        for i in range(1, len(indices)):
            index += indices[i] * factor
            factor *= self.limits[i]
        return index

    def get_indices(self, index: int):
        indices = []
        for i in range(len(self.limits)):
            indices.append(index % self.limits[i])
            index //= self.limits[i]
        return tuple(indices)

    def flatten_array(self, array: np.ndarray):
        flattened = np.zeros(self.length, dtype=array.dtype)
        for i in range(self.length):
            flattened[i] = array[self.get_indices(i)]
        return flattened

    def unflatten_list(self, flattened: np.ndarray):
        array = np.zeros(self.limits, dtype=flattened.dtype)
        for i in range(self.length):
            array[self.get_indices(i)] = flattened[i]
        return array

    def __iter__(self):
        for i in range(self.length):
            yield self.get_indices(i)

    def __len__(self):
        return self.length

    def __call__(self, arg):
        if type(arg) == int:
            return self.get_indices(arg)
        elif type(arg) == tuple:
            return self.get_index(arg)
        else:
            raise TypeError(f'Type of arg must be int or tuple, but {type(arg)} was given!')


def main():
    limits = (3, 3, 3, 3)
    converter = MultiIndex(limits)
    print(converter(0))
    print(converter(1))
    print(converter(2))
    print(converter(3))

# array = np.zeros(limits)
    #
    # for i in range(converter.length):
    #     array[converter.get_indices(i)] = i
    #
    # print(array)
    # print(converter.flatten_array(array))
    # print(converter.unflatten_list(converter.flatten_array(array)))
    #
    # for i in converter:
    #     print(i)
    #
    # print(converter((2, 2, 2)))
    # print(converter(5))
    # print(converter(-1))
    # print(converter(-len(converter)))


if __name__ == '__main__':
    main()
