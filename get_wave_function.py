from CantorSchrodinger import Solver
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():
    for i in range(4):
        solver = Solver(i)
        state = np.load(f'./groundstate{i}.npy')
        f = solver.get_wave_function(state)
        domain = np.linspace(0, 1)
        df = pd.DataFrame(columns=['x', 'y', 'wave function'])
        for x in domain:
            for y in domain:
                df.loc[len(df)] = [x, y, f(x, y)]
        df.to_csv(f'wave_function{i}.txt', sep=' ', index=False)
        xx, yy = np.meshgrid(domain, domain, indexing='xy')
        plt.figure()
        plt.contourf(xx, yy, np.reshape(df['wave function'].to_numpy(), xx.shape))
        plt.colorbar()
        plt.savefig(f'wave_function{i}.pdf')


if __name__ == '__main__':
    main()
