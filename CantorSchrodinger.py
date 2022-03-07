import numpy as np
from MultiIndex import MultiIndex
from SymmetricIndex import SymmetricIndex
from AntiSymmetricIndex import AntiSymmetricIndex
from CantorSet import CantorSet
import scipy.sparse
import scipy.sparse.linalg


class Solver:
    def __init__(self, depth: int, repulsion_strength: float = 10, excited_states: int = 10,
                 interaction: str = 'harmonic', electric_field: float = .1, symmetry=None):
        # recursion depth of the cantor set
        self.depth = depth
        # number of excited states of square well to consider
        self.excited_states = excited_states
        # length of a single square well
        self.length = 1 / 3 ** depth
        # strength V0 of the interaction potential (V(x,y) = V0 * (1 - |x - y|) or V(x,y) = V0 * (x - y) ^ 2)
        self.interaction_strength = repulsion_strength
        self.electric_field = electric_field
        # edges of the cantor set
        self.edges = CantorSet(depth)
        if symmetry is None:
            # limits for the indices
            self.limits = (len(self.edges), len(self.edges), excited_states, excited_states)
            indices = MultiIndex(self.limits)
            self.indices = indices
            # init hamiltonian matrix
            self.hamiltonian = scipy.sparse.lil_matrix((len(indices), len(indices)))
            for I in range(len(indices)):
                # iteration over all indices takes too long for large matrices
                # iterations is only needed over states where wave functions of the same variable are in the same well
                i, j, n1, m1 = indices(I)
                for n2 in range(excited_states):
                    for m2 in range(excited_states):
                        J = indices((i, j, n2, m2))
                        # calculation of the interaction integral
                        self.hamiltonian[I, J] = self.interaction_integral((i, j, n1, m1), (i, j, n2, m2),
                                                                           interaction=interaction)
                        # add unperturbed energies in the diagonal terms (I = J)
                        if m1 == m2 and n1 == n2:
                            self.hamiltonian[I, J] += self.unperturbed_energy(n1) + self.unperturbed_energy(m1)
                        # if particle == 'fermion':
                        #     self.hamiltonian[I, J] -= self.interaction_integral((i, j, n1, m1), (i, j, m2, n2))
                        #     if n1 == m2 and m1 == n2:
                        #         self.hamiltonian[I, J] -= self.unperturbed_energy(n1) + self.unperturbed_energy(n2)
        if symmetry == 'symmetric':
            particle_index = MultiIndex((len(self.edges), excited_states))
            combined_index = SymmetricIndex(len(particle_index))
            self.hamiltonian = scipy.sparse.lil_matrix((len(combined_index), len(combined_index)))
            for I, (a, b) in combined_index:
                i, n1 = particle_index(a)
                j, m1 = particle_index(b)
                for n2 in range(excited_states):
                    for m2 in range(excited_states):
                        J = combined_index(particle_index((i, n2)), particle_index((j, m2)))
                        self.hamiltonian[I, J] = self.interaction_integral((i, j, n1, m1), (i, j, n2, m2))
                        if I == J:
                            self.hamiltonian[I, J] += self.unperturbed_energy(n1) + self.unperturbed_energy(m2)
                        self.hamiltonian[I, J] += self.interaction_integral((i, j, n1, m1), (j, i, m2, n2))
                        if n1 == m2 and m1 == n2 and i == j:
                            self.hamiltonian[I, J] += self.unperturbed_energy(n1) + self.unperturbed_energy(n2)
                        # handle the cases of same state as, |a,b> = (|a>|b>+|b>|a>)/sqrt(2), but |a,a>=|a>|a>
                        if a == b:
                            self.hamiltonian /= np.sqrt(2)
                        if (i, n2) == (j, m2):
                            self.hamiltonian /= np.sqrt(2)
        if symmetry == 'antisymmetric':
            particle_index = MultiIndex((len(self.edges), excited_states))
            combined_index = AntiSymmetricIndex(len(particle_index))
            self.hamiltonian = scipy.sparse.lil_matrix((len(combined_index), len(combined_index)))
            for I, (a, b) in combined_index:
                i, n1 = particle_index(a)
                j, m1 = particle_index(b)
                for n2 in range(excited_states):
                    for m2 in range(excited_states):
                        J = combined_index(particle_index((i, n2)), particle_index((j, m2)))
                        self.hamiltonian[I, J] = self.interaction_integral((i, j, n1, m1), (i, j, n2, m2))
                        if I == J:
                            self.hamiltonian[I, J] += self.unperturbed_energy(n1) + self.unperturbed_energy(m2)
                        self.hamiltonian[I, J] -= self.interaction_integral((i, j, n1, m1), (j, i, m2, n2))
                        if n1 == m2 and m1 == n2 and i == j:
                            self.hamiltonian[I, J] -= self.unperturbed_energy(n1) + self.unperturbed_energy(n2)
            # print(f'{(I + 1) * len(indices) / len(indices) ** 2 * 100:.2f}% done!')
        # print('Init done!')

    def unperturbed_energy(self, n: int):
        return np.pi ** 2 / (2 * self.length ** 2) * (n + 1) ** 2

    def interaction_integral(self, indices1: tuple, indices2: tuple, interaction: str = 'harmonic'):
        V = self.interaction_strength
        E = self.electric_field
        i1, j1, n1, m1 = indices1
        i2, j2, n2, m2 = indices2
        if i1 != i2 or j1 != j2:
            return 0
        i, j = i1, j1
        n1, n2, m1, m2 = n1 + 1, n2 + 1, m1 + 1, m2 + 1
        # Result for V(x,y) = V (1 - |x-y|)
        if interaction == 'repulsive':
            if n1 != n2 and m1 != m2:
                return 0
            if i != j:
                if n1 == n2:
                    if m1 == m2:
                        return V * (1 - np.abs(self.edges[i][0] - self.edges[j][0]))
                    else:
                        return V * np.sign(i - j) * 4 * self.length / np.pi ** 2 * m1 * m2 * (
                                -1 + (-1) ** (m1 + m2)) / (
                                       m1 ** 2 - m2 ** 2) ** 2
                else:
                    return -V * np.sign(i - j) * 4 * self.length / np.pi ** 2 * n1 * n2 * (-1 + (-1) ** (n1 + n2)) / (
                            n1 ** 2 - n2 ** 2) ** 2
            if i == j:
                if n1 == n2:
                    if m1 == m2:
                        return V * (1 + self.length / 6 * (2 - 3 / np.pi ** 2 * (1 / n1 ** 2 + 1 / m1 ** 2)))
                    else:
                        return V * 4 * self.length / np.pi ** 2 * m1 * m2 * (1 + (-1) ** (m1 + m2)) / (
                                m1 ** 2 - m2 ** 2) ** 2
                else:
                    return V * 4 * self.length / np.pi ** 2 * n1 * n2 * (1 + (-1) ** (n1 + n2)) / (
                            n1 ** 2 - n2 ** 2) ** 2
        # Result for V(x,y) = V * (x - y)^2 + E * (x + y)
        elif interaction == 'harmonic':
            i, j = self.edges[i][0], self.edges[j][0]
            n, m = n1, n2
            k, l = m1, m2
            L = self.length
            if n1 != n2 and m1 != m2:
                return -32 * V * n1 * n2 * m1 * m2 * self.length ** 2 * (-1 + (-1) ** (m1 + m2)) * (
                        -1 + (-1) ** (n1 + n2)) / ((n1 ** 2 - n2 ** 2) ** 2 * (m1 ** 2 - m2 ** 2) ** 2 * np.pi ** 4)
            elif n1 == n2 and m1 != m2:
                # return 4 * V * self.length * m1 * m2 * (
                #         2 * (i - j) + self.length + (-1) ** (m1 + m2) * (2 * (j - i) + self.length)) / (
                #                (m1 ** 2 - m2 ** 2) ** 2 * np.pi ** 2)
                return 4 * k * l * L * (
                        -E + 2 * V * i - 2 * V * j + V * L + (-1) ** (k + l) * (E + V * (-2 * i + 2 * j + L))) / (
                               (k ** 2 - l ** 2) ** 2 * np.pi ** 2)
            elif n1 != n2 and m1 == m2:
                # return 4 * V * self.length * n1 * n1 * (
                #         2 * (j - i) + self.length + (-1) ** (n1 + n2) * (2 * (i - j) + self.length)) / (
                #                (n1 ** 2 - n2 ** 2) ** 2 * np.pi ** 2)
                return 4 * L * m * n * (
                        -E - 2 * V * i + 2 * V * j + V * L + (-1) ** (m + n) * (E + V * (2 * i - 2 * j + L))) / (
                               (m ** 2 - n ** 2) ** 2 * np.pi ** 2)
            else:
                # return V * (i - j) ** 2 + V * self.length ** 2 / 6 * (
                #         1 - 3 / (m1 ** 2 * np.pi ** 2) - 3 / (n1 ** 2 * np.pi ** 2))
                return E * (i + j + L) + V / 6 * (
                        6 * (i - j) ** 2 + L ** 2 - 3 * L ** 2 * (k ** 2 + n ** 2) / (k ** 2 * n ** 2 * np.pi ** 2))
        else:
            raise ValueError(f'{interaction=} not supported')

    def get_eigenvalues(self):
        return scipy.sparse.linalg.eigsh(self.hamiltonian, min(len(self.hamiltonian.rows) - 1, 1000), which='SM')

    def get_well_function(self, i, n, samples: int = 200):
        def f(arr):
            result = np.zeros_like(arr)
            for j in range(len(arr)):
                x = arr[j]
                if self.edges[i][0] < x < self.edges[i][1]:
                    result[j] = np.sqrt(2 / self.length) * np.sin(
                        (n + 1) * np.pi / self.length * (x - self.edges[i][0]))
            return result

        return f(np.linspace(0, 1, samples))

    def get_wave_function(self, state: np.ndarray):
        c1 = np.sqrt(2 / self.length)
        c2 = np.pi / self.length

        def well_function(i, n):
            def f(x):
                if self.edges[i][0] < x < self.edges[i][1]:
                    return c1 * np.sin((n + 1) * c2 * (x - self.edges[i][0]))
                else:
                    return 0

            return f

        def f(x, y):
            s = 0
            for I in range(len(self.indices)):
                if np.abs(state[I]) > 1e-8:
                    i, j, n, m = self.indices(I)
                    s += state[I] * well_function(i, n)(x) * well_function(j, m)(y)
            return s

        return f


def main():
    import matplotlib.pyplot as plt
    import scipy.integrate
    from tqdm import trange

    max_depth = 5
    load = False
    samples = 200
    groundstates = []
    for depth in trange(max_depth):
        if not load:
            solver = Solver(depth, interaction='harmonic', electric_field=1.0, symmetry='symmetric')
            energys, states = solver.get_eigenvalues()
            np.save(f'energy{depth}', energys)
            np.savetxt(f'energy{depth}.txt', energys)
            # for plotting
            groundstate = states[:, 0]
            np.save(f'groundstate{depth}', groundstate)
            # indices = solver.indices
            # particle1 = np.zeros(samples)
            # particle2 = np.zeros(samples)
            # for I in range(len(groundstate)):
            #     i, j, n, m = indices(I)
            #     particle1 += groundstate[I] * solver.get_well_function(i, n, samples)
            #     particle2 += groundstate[I] * solver.get_well_function(j, m, samples)
            # groundstates.append([particle1, particle2])
            groundstates.append(solver.get_wave_function(groundstate))
        else:
            energys = np.load(f'energy{depth}.npy')
            np.savetxt(f'energy{depth}.txt', energys)

        # print(len(energys))
        # indices = solver.indices
        # print(energys[:])
        # print(states)
        # print(indices.unflatten_list(states[:, 0]))

        plt.scatter(energys / 9 ** depth, [depth] * len(energys), label=f'depth = {depth}', s=2)
    plt.legend()
    plt.show()
    # if not load:
    #     x = np.linspace(0, 1, samples)
    #     for wave_function in groundstates:
    #         def density(x):
    #             def f(y):
    #                 w = wave_function(x, y)
    #                 return w * w
    #
    #             return scipy.integrate.quad(f, 0, 1)
    #
    #         plt.plot(x, [density(_) for _ in x])
    #         plt.show()


if __name__ == '__main__':
    main()
