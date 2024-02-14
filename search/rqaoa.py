import os
import sys
import time
import itertools
from typing import Optional, List, Tuple, Callable

import numpy as np
import numba
from numba import jit
import pandas as pd

import networkx as nx
import symengine as sym

import scipy
from scipy import optimize
from scipy.spatial.distance import hamming

from tqdm import tqdm
import pprint
import pickle

np.set_printoptions(suppress=True)


def karloff(m, t, b):
    # ideally one would want to choose b s.t. its value is follows (m - 2b)/m = cos(\theta), where \theta = 2.331122
    # creating a list of t-element subsets of a set {1,...,n} = [n]
    # there has been some hand waving results...for more info check out Swati Gupta's new warm start QAOA paper:https://arxiv.org/pdf/2112.11354.pdf)
    vertices = list(itertools.combinations(np.arange(1, m + 1), t))
    G = nx.Graph()
    for i in range(len(vertices)):
        for j in range(i + 1, len(vertices)):
            if len(list(set(vertices[i]) & set(vertices[j]))) == b:
                G.add_edge(i, j)
    return G


def karloff_alon(m, b):
    vert_cand = itertools.product([-1, 1], repeat=m)  # generates 2^m bitstrings
    bs = []
    for i in vert_cand:
        bs.append(i)
    G = nx.Graph()
    #     for i in range(len(bs)):
    #         for j in range(i+1, len(bs)):
    #             if np.inner(bs[i], bs[j]) == 1 - 2*c:
    #                 G.add_edge(i, j)
    for i in range(len(bs)):
        for j in range(i + 1, len(bs)):
            if hamming(bs[i], bs[j]) * len(bs[i]) == b:
                G.add_edge(i, j)
    return G


def random_weights(graph: nx.Graph,
                   rs: Optional[np.random.RandomState] = None,
                   type: str = 'bimodal',
                   weight_type=None,
                   min_weight=None,
                   max_weight=None):
    """Take a graph and make an equivalent graph with weights of plus or minus
    one on each edge.
    Args:
    graph: A graph to add weights to
    rs: A RandomState for making replicable experiments. If not provided,
        the global numpy random state will be used. Please be careful
        when generating random problems. You should construct *one*
        seeded RandomState at the beginning of your script and use
        that one RandomState (in a deterministic fashion) to generate
        all the problem instances you may need.
    """

    if rs is None:
        rs = np.random
    elif not isinstance(rs, np.random.RandomState):
        raise ValueError("Invalid random state: {}".format(rs))

    problem_graph = nx.Graph()
    for n1, n2 in graph.edges:
        if type == 'bimodal':
            problem_graph.add_edge(n1, n2, weight=rs.choice([-1, 1]))
        elif type == 'pos_gaussian':
            problem_graph.add_edge(n1, n2, weight=np.abs(rs.randn()))
        elif type == 'gaussian':
            problem_graph.add_edge(n1, n2, weight=rs.randn())
        elif type == 'sidon_567':
            problem_graph.add_edge(n1, n2, weight=rs.choice([-5, -6, -7, 5, 6, 7]))
        elif type == 'sidon_28_normalized':
            problem_graph.add_edge(n1, n2,
                                   weight=rs.choice([-8 / 28, -13 / 28, -19 / 28, -1, 8 / 28, 13 / 28, 19 / 28, 1]))
        elif type == 'one':
            problem_graph.add_edge(n1, n2, weight=rs.choice([1]))
        elif type == 'custom':
            if weight_type == 'float':
                problem_graph.add_edge(n1, n2, weight=rs.uniform(min_weight, max_weight))
            elif weight_type == 'integer':
                problem_graph.add_edge(n1, n2, weight=rs.choice(np.arange(min_weight, max_weight + 1)))
    return problem_graph


class Graph:
    def __init__(self,
                 n: int,
                 d: int = None,
                 G: nx.Graph = None):
        self.n = n
        self.d = d
        if G is None:
            G = nx.generators.random_graphs.random_regular_graph(d, n, seed=42)
            G = random_weights(graph=G, rs=np.random.RandomState(42))
        elif G is 'sk_problem':
            G = nx.complete_graph(n)
            G = random_weights(graph=G, rs=np.random.RandomState(42))
        self.G = G
        self.G0 = G.copy()

    def reset(self):
        self.G = self.G0.copy()

    def get_G_numpy(self,
                    nodelist: List[int] = None):
        if nodelist is None:
            nodelist = range(self.n)
        return nx.to_numpy_array(self.G, dtype=np.float, nodelist=nodelist)

    def get_G_sparse(self,
                     nodelist: List[int] = None):
        if nodelist is None:
            nodelist = range(self.n)
        return nx.to_scipy_sparse_matrix(self.G, dtype=np.float, nodelist=nodelist)

    def eliminate(self,
                  edge: Tuple[int],
                  sign: float):
        rmv_edges = []
        updt_edges = []
        add_edges = []
        for neighb in self.G.neighbors(edge[1]):
            rmv_edges += [(edge[1], neighb)]
            if neighb not in edge:
                if (edge[0], neighb) in self.G.edges():
                    self.G[edge[0]][neighb]['weight'] += sign * self.G[edge[1]][neighb]['weight']
                    if self.G[edge[0]][neighb]['weight'] == 0:
                        self.G.remove_edge(edge[0], neighb)
                        rmv_edges += [(edge[0], neighb)]
                    else:
                        updt_edges += [(edge[0], neighb)]
                else:
                    self.G.add_edge(edge[0], neighb, weight=sign * self.G[edge[1]][neighb]['weight'])
                    add_edges += [(edge[0], neighb)]
        for e in self.G.edges(edge[0]):
            if e not in updt_edges + rmv_edges + add_edges:
                updt_edges += [e]
        for neighb in self.G.neighbors(edge[1]):
            for e in self.G.edges(neighb):
                if e not in updt_edges + rmv_edges + add_edges:
                    updt_edges += [e]
        self.G.remove_node(edge[1])
        return rmv_edges, updt_edges, add_edges


class RQAOA:

    def __init__(self,
                 n: int,
                 nc: int,
                 d: int,
                 batch_size: int,
                 grid_N: int,
                 search_space: List,
                 G: nx.Graph = None,
                 solver: str = 'analytic_brute'):

        self.n = n
        self.nc = nc
        self.graph = Graph(n, d, G)
        self.batch_size = batch_size
        self.w = self.graph.get_G_numpy()
        self.all_angles = []
        self.ref = ()
        self.solver = solver
        self.grid_N = grid_N
        self.search_space = search_space

        self.x = sym.Symbol('x', real=True)
        self.y = sym.Symbol('y', real=True)

    def run_rqaoa(self):
        batch_rqaoa_angles, batch_energies, batch_zs, exp_flags, ties_flags, ties_batch = [], [], [], [], [], []
        for i in range(self.batch_size):
            print("RQAOA agent " + str(i + 1) + "/" + str(self.batch_size))
            rqaoa_angles, energy, z, te, exp_flag, ties_flag, check_ties = self.rqaoa()
            exp_flags += [exp_flag]
            ties_flags += [ties_flag]
            batch_rqaoa_angles += [rqaoa_angles]
            batch_energies += [energy]
            batch_zs += [z]
            ties_batch += [check_ties]
            print("Energy: " + str(energy))
        arg = np.argmax(batch_energies)
        rqaoa_angles = np.array(batch_rqaoa_angles[arg], dtype=float)
        ref = (batch_energies[arg], batch_zs[arg], batch_energies)
        self.all_angles = rqaoa_angles
        self.ref = ref
        # print(ties_batch)
        return batch_energies[arg], batch_zs[
            arg], rqaoa_angles, batch_energies, batch_zs, batch_rqaoa_angles, te, exp_flags, ties_flags, ties_batch

    def rqaoa(self):
        self.graph.reset()
        nodelist = np.arange(self.n)
        J = self.graph.get_G_numpy(nodelist)

        f_s, h, action_space = self.generate_fs_h_actions(J, nodelist)
        assignments = []
        signs = []
        check_exp = []
        check_ties = []
        rqaoa_angles = np.array([], dtype=float)
        ts = time.time()
        for m in tqdm(range(self.n - self.nc)):
            angles, f_val = self.compute_extrema(h, solver=self.solver, search_space=self.search_space)
            # angles = solutions[np.argmax(extrema)]
            rqaoa_angles = np.append(rqaoa_angles, angles)
            expectations, indcs = self.compute_expectations(f_s, angles)
            # print(expectations)
            if len(expectations) == len(set(expectations)):  # check if edge correlations are unique
                check_exp.append(m)
            abs_expectations = np.abs(expectations)
            max_abs = np.flatnonzero(abs_expectations == abs_expectations.max())
            # print(f'Maximal Two-correlations: {np.max(abs_expectations)}')
            # print(f'Two correlations: {max_abs}')
            # print(f'Same #Correlations  - {len(max_abs)} at Iteration {m}')
            check_ties.append(len(max_abs))
            idx = np.random.choice(max_abs)
            # print(idx)
            edge, sign = action_space[idx], np.sign(expectations[idx])
            # print(edge, sign)
            rmv_edges, updt_edges, add_edges = self.graph.eliminate(edge, sign)
            assignments += [edge]
            signs += [sign]
            nodelist = nodelist[nodelist != edge[1]]
            J = self.graph.get_G_numpy(nodelist)
            f_s, action_space = self.update(f_s, action_space, J, nodelist, rmv_edges, updt_edges, add_edges)
            h = self.compute_h(f_s, action_space, J, nodelist)

        # print(f'after for loop {repr(J)}')
        _, z_c, z_ss = self.bruteforce_full_instance(J, self.nc)
        # print(f'ZSS = {z_ss}')
        # print(nodelist)
        z_s, ep_energies = self.expand_result(z_c, assignments, signs, nodelist)
        print(time.time() - ts)
        te = time.time() - ts

        if check_exp == list(np.arange(self.n - self.nc)):
            exp_flag = 1  # if edge correlations at all iterations are unique when wt's are N(0,1)
        else:
            exp_flag = 0

        if len(check_ties) == np.sum(check_ties):
            ties_flag = 1  # if there are no ties while using argmax
        else:
            ties_flag = 0

        return rqaoa_angles, ep_energies[0], z_s[0], te, exp_flag, ties_flag, check_ties

    def store_agent(self, pickle_path=None):
        if pickle_path is None:
            pickle_path = self.pickle_path
        pickle.dump(self, open(pickle_path, 'wb'))

    def update(self,
               f_s,
               action_space: List[Tuple[int]],
               J: np.ndarray,
               nodelist: List[int],
               rmv_edges: List[Tuple[int]],
               updt_edges: List[Tuple[int]],
               add_edges: List[Tuple[int]]):
        nl = list(nodelist)
        for edge in rmv_edges:
            if edge[0] > edge[1]:
                edge = (edge[1], edge[0])
            indx = action_space.index(edge)
            action_space.pop(indx)
            f_s.pop(indx)

        for edge in add_edges:
            if edge[0] > edge[1]:
                edge = (edge[1], edge[0])
            inserted = False
            for i in range(len(action_space)):
                edge_i = action_space[i]
                if (edge_i[0] == edge[0] and edge_i[1] > edge[1]) or edge_i[0] > edge[0]:
                    action_space.insert(i, edge)
                    f_s.insert(i, self.compute_f(J, nl.index(edge[0]), nl.index(edge[1])))
                    inserted = True
                    break
            if not inserted:
                action_space += [edge]
                f_s += [self.compute_f(J, nl.index(edge[0]), nl.index(edge[1]))]

        for edge in updt_edges:
            if edge[0] > edge[1]:
                edge = (edge[1], edge[0])
            if edge in action_space:
                indx = action_space.index(edge)
                f_s[indx] = self.compute_f(J, nl.index(edge[0]), nl.index(edge[1]))

        return f_s, action_space

    def compute_h(self,
                  f_s,
                  action_space: List[Tuple[int]],
                  J: np.ndarray,
                  nodelist: List[int]):
        h = 0.
        count = 0
        for i in range(len(J)):
            for j in range(i + 1, len(J)):
                if J[i, j]:
                    if action_space[count] != (nodelist[i], nodelist[j]):
                        print("Wrong count")
                    h += J[i, j] * f_s[count]  # max-cut

                    count += 1
        return h

    def compute_expectations(self,
                             f_s,
                             angles: List[float]):
        expectations = []
        indcs = []
        x = self.x
        y = self.y

        for i, f in enumerate(f_s):
            if f in f_s[:i]:
                indx = f_s.index(f)
                expectations += [expectations[indx]]
                indcs += [indx]
            else:
                x0, y0 = angles[0], angles[1]
                expectations += [float(f.subs({x: x0, y: y0}))]
                indcs += [i]

        return expectations, indcs

    def get_binary(self,
                   x: int,
                   n: int):

        return 2 * np.array([int(b) for b in bin(x)[2:].zfill(n)], dtype=np.int32) - 1

    def bruteforce(self,
                   J: np.ndarray,
                   n: int):
        maxi = -n
        idx = []
        for i in range(2 ** n):
            z = self.get_binary(i, n)
            val = 0
            for k in range(len(J)):
                for l in range(k + 1, len(J)):
                    if z[k] != z[l]:    # if the edges are in different partition
                        val += J[k, l]
            if val > maxi:
                maxi = val
                idx = [i]
            elif val == maxi:
                idx += [i]
        return maxi, idx

    def bruteforce_full_instance(self,
                                 J: np.ndarray,
                                 n: int):
        max_cut, idx = self.bruteforce(J, n)
        zs = []
        for i in idx:
            zs.append(self.get_binary(i, n))
        return max_cut, idx, zs

    def compute_f(self,
                  J: np.ndarray,
                  i: int,
                  j: int):

        x, y = self.x, self.y
        C = sym.cos
        S = sym.sin

        prod1, prod2, prod3, prod4 = 1., 1., 1., 1.
        for k in range(len(J)):
            if k not in [i, j]:
                if J[i, k] - J[j, k]:
                    prod1 *= C(2 * x * (J[i, k] - J[j, k]))
                if J[i, k] + J[j, k]:
                    prod2 *= C(2 * x * (J[i, k] + J[j, k]))
                if J[i, k]:
                    prod3 *= C(2 * x * J[i, k])
                if J[j, k]:
                    prod4 *= C(2 * x * J[j, k])
        term = 0.5 * (S(2 * y) ** 2) * (prod1 - prod2) + 0.5 * S(4 * y) * S(2 * x * J[i, j]) * (prod3 + prod4)

        return term

    def generate_fs_h_actions(self,
                              J: np.ndarray,
                              nodelist: List[int]):
        f_s = []
        h = 0.
        action_space = []

        for i in range(len(J)):
            for j in range(i + 1, len(J)):
                if J[i, j]:
                    action_space += [(nodelist[i], nodelist[j])]
                    term = self.compute_f(J, i, j)
                    f_s += [term]
                    h += J[i, j] * term   # for max-cut
        return f_s, h, action_space

    def compute_extrema(self,
                        h,
                        search_space: List,
                        solver: str = 'analytic_brute',
                        polishing_optimizer: Callable = optimize.cobyla):

        x = self.x
        y = self.y
        # print(h)
        if solver == 'analytic_brute':
            # \gamma is x and \beta is y
            # E(x, y) = - (p * cos(4 * y) + q * sin(4 * y) + r), where p,q,r are complicated eqn's of x. Find p,q,r.
            # A -ve sign because of max cut
            r = - (h.subs({x: x, y: np.pi / 8}) + h.subs({x: x, y: -np.pi / 8})) / 2
            q = - (h.subs({x: x, y: np.pi / 8}) - h.subs({x: x, y: -np.pi / 8})) / 2
            p = - h.subs({x: x, y: 0}) - r

            # No -ve sign for fun as we want to minimize the energy that gives us the max cut
            max_y_fun = - (r + sym.sqrt(p ** 2 + q ** 2))  # maximum of E(x,y) over all y's.
            fun = sym.Lambdify([(x)], max_y_fun, backend='llvm')

            # for 3-reg landscape is symmetrical for gamma between 0 to np.pi at np.pi/2
            param_ranges = (slice(search_space[0], search_space[1], abs(search_space[1] - search_space[0]) / grid_N),)
            res_brute = optimize.brute(fun, param_ranges, full_output=True, finish=polishing_optimizer)

            solution = [res_brute[0]]

            q_val = q.subs({x: solution[0]})
            p_val = p.subs({x: solution[0]})
            y_val = 1 / 4 * (sym.atan2(q_val, p_val))

            assert (p_val * sym.cos(4 * y_val) >= 0)
            assert (q_val * sym.sin(4 * y_val) >= 0)
            solution = np.append(solution, np.float(y_val))
            extrema = res_brute[1]
            # print(solution, extrema)
        else:
            raise ValueError(f'Optimizer {self.solver}  is not implemented')

        return solution, extrema

    def expand_result(self,
                      z_c: List[int],
                      assignments: List[Tuple[int]],
                      signs: List[int],
                      nodelist: List[int]):

        z_s = [self.get_binary(z, self.nc) for z in z_c]
        # print(z_s)
        # print('---------')
        z_s = [np.array([z[nodelist.tolist().index(i)] if i in nodelist else 0 for i in range(self.n)], dtype=np.int32)
               for z in z_s]
        # print(z_s)
        # print('--------')
        # print(signs)
        self.graph.reset()
        J = self.graph.get_G_numpy()

        ep_energies = []
        for i, assgn in enumerate(assignments[::-1]):
            for j, z in enumerate(z_s):
                z[assgn[1]] = signs[-i - 1] * z[assgn[0]]
                z_s[j] = z

            val = 0
            for k in range(len(J)):
                for l in range(k + 1, len(J)):
                    if z_s[0][k] != z_s[0][l]:    # if the edges are in different partition
                        val += J[k, l]
            ep_energies.insert(0, val)
        return z_s, ep_energies


if __name__ == '__main__':
    # TODO: smallest worst case instance for RQAOA (n=9, m=24), best ising=26 and whatever n_c you put best_rqaoa=18
    # approx ratio = 18/26 = 0.6923076923
    # J = [[0., -1., 3., 0., 0., -1., 0., 1., -2.],
    #      [-1., 0., 1., 0., 1., 1., -1., -1., 0.],
    #      [3., 1., 0., 1., 0., 0., 1., -1., 1.],
    #      [0., 0., 1., 0., 1., 0., 0., 1., 1.],
    #      [0., 1., 0., 1., 0., -1., 0., 0., 1.],
    #      [-1., 1., 0., 0., -1., 0., 1., 1., 1.],
    #      [0., -1., 1., 0., 0., 1., 0., 0., 1.],
    #      [1., -1., -1., 1., 0., 1., 0., 0., -1.],
    #      [-2., 0., 1., 1., 1., 1., 1., -1., 0.]]

    # df = pd.DataFrame(J)
    # G = nx.convert_matrix.from_pandas_adjacency(df)
    # n = 4  # Number of nodes in graph
    # G = nx.Graph()
    # G.add_nodes_from(np.arange(0, n, 1))
    # elist = [(0, 1, 1.0), (0, 2, 1.0), (0, 3, 1.0), (1, 2, 1.0), (2, 3, 1.0)]
    # # tuple is (i,j,weight) where (i,j) is the edge
    # G.add_weighted_edges_from(elist)
    # example_graph_dict = {
    #     0: {1: {"weight": 1.0}, 7: {"weight": 1.0}, 3: {"weight": 1.0}},
    #     1: {0: {"weight": 1.0}, 2: {"weight": 1.0}, 3: {"weight": 1.0}},
    #     2: {1: {"weight": 1.0}, 3: {"weight": 1.0}, 5: {"weight": 1.0}},
    #     3: {1: {"weight": 1.0}, 2: {"weight": 1.0}, 0: {"weight": 1.0}},
    #     4: {7: {"weight": 1.0}, 6: {"weight": 1.0}, 5: {"weight": 1.0}},
    #     5: {6: {"weight": 1.0}, 4: {"weight": 1.0}, 2: {"weight": 1.0}},
    #     6: {7: {"weight": 1.0}, 4: {"weight": 1.0}, 5: {"weight": 1.0}},
    #     7: {4: {"weight": 1.0}, 6: {"weight": 1.0}, 0: {"weight": 1.0}},
    # }
    #
    # G = nx.to_networkx_graph(example_graph_dict)

    weighted_graph_dict = {
        0: {1: {"weight": 0.1756}, 7: {"weight": 2.5664}, 3: {"weight": 1.8383}},
        1: {0: {"weight": 0.1756}, 2: {"weight": 2.2142}, 3: {"weight": 4.7169}},
        2: {1: {"weight": 2.2142}, 3: {"weight": 2.0984}, 5: {"weight": 0.1699}},
        3: {1: {"weight": 4.7169}, 2: {"weight": 2.0984}, 0: {"weight": 1.8383}},
        4: {7: {"weight": 0.9870}, 6: {"weight": 0.0480}, 5: {"weight": 4.2509}},
        6: {7: {"weight": 4.7528}, 4: {"weight": 0.0480}, 5: {"weight": 2.2879}},
        5: {6: {"weight": 2.2879}, 4: {"weight": 4.2509}, 2: {"weight": 0.1699}},
        7: {4: {"weight": 0.9870}, 6: {"weight": 4.7528}, 0: {"weight": 2.5664}},
    }

    G = nx.to_networkx_graph(weighted_graph_dict)

    grid_N, search_space, solver, batch_size = 100, [0, 2 * np.pi], 'analytic_brute', 1
    J = nx.to_numpy_array(G)
    # G = G.networkx_graph()

    n = G.number_of_nodes()
    r = RQAOA(n=n, nc=3, d=None, G=G, batch_size=batch_size, solver=solver, grid_N=grid_N, search_space=search_space)

    print(r.bruteforce_full_instance(J, n))
    batch_energies1, batch_zs1, rqaoa_angles, batch_energies, batch_zs, batch_rqaoa_angles, te, exp_flags, ties_flags, ties_batch = r.run_rqaoa()
    print(ties_batch, ties_flags, batch_energies1, batch_zs1)




