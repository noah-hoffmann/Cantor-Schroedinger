class CantorSet:
    def __init__(self, depth: int):
        if depth < 0:
            raise ValueError('depth must be a non negative integer')
        self.depth = depth
        self.edges = CantorSet.edge_points(depth)

    @staticmethod
    def edge_points(depth: int, points: list = None):
        if points is None:
            points = [(0., 1.)]
            return CantorSet.edge_points(depth - 1, points)
        if depth < 0:
            return points
        else:
            new_points = []
            for a, b in points:
                p1 = (2 * a + b) / 3
                p2 = (a + 2 * b) / 3
                new_points.append((a, p1))
                new_points.append((p2, b))
            return CantorSet.edge_points(depth - 1, new_points)

    def __iter__(self):
        return self.edges.__iter__()

    def __len__(self):
        return len(self.edges)

    def __getitem__(self, item):
        return self.edges[item]


def main():
    import matplotlib.pyplot as plt

    for i in range(10):
        points = CantorSet(i)
        print(len(points))
        for p in points:
            plt.plot(p, [i * 0.2, i * 0.2], 'r')
    plt.show()


if __name__ == '__main__':
    main()
