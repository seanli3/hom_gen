import copy
import itertools
import hashlib
import networkx as nx
from collections import Counter, defaultdict


# credit: https://towardsdatascience.com/testing-if-two-graphs-are-isomorphic-cf6c44ab551e

def base_WL(verbose, n_set, initial_colors_func, find_neighbors_func, iterations=None):
    G, n = n_set()
    colors = initial_colors_func(n)
    recorded_node_canonical_forms = []
    recorded_canonical_forms = []
    old_colors = copy.deepcopy(colors)
    for i in range(iterations if iterations is not None else len(G)):
        recorded_node_canonical_forms.append([])
        for node in n:
            sorted_neigh_colors = sorted([colors[i][0] for i in find_neighbors_func(G, n, node)])
            neigh_colors = "".join(sorted_neigh_colors)
            recorded_node_canonical_forms[i].append([*colors[node], *sorted_neigh_colors])
            colors[node].extend([neigh_colors])
        for node in n:
            recorded_node_canonical_forms[i][node] = sorted(Counter(recorded_node_canonical_forms[i][node]).items())
        # Update with the hash
        colors = {i: [hashlib.sha224("".join(colors[i]).encode('utf-8')).hexdigest()] for i in colors}

        if list(Counter([item for sublist in colors.values() for item in sublist]).values()) == list(
                Counter([item for sublist in old_colors.values() for item in sublist]).values()) and i != 0:
            recorded_node_canonical_forms.pop()
            if verbose:
                print(f'Converged at iteration {i}!')
            break
        old_colors = copy.deepcopy(colors)
        canonical_form = sorted(Counter([item for sublist in colors.values() for item in sublist]).items())
        recorded_canonical_forms.append(canonical_form)
    if verbose:
        print(f'Canonical Form Found: \n {canonical_form} \n')
        # print(f'Iterations: {i} \n')
    return canonical_form, recorded_node_canonical_forms, recorded_canonical_forms


def WL(G, verbose=False, iterations=None):
    G = nx.convert_node_labels_to_integers(G)

    def n_set():
        return G, list(G.nodes())

    def set_initial_colors(n):
        return {i: [hashlib.sha224("1".encode('utf-8')).hexdigest()] for i in n}

    def find_neighbors(G, n, node):
        return G.neighbors(node)

    return base_WL(verbose, n_set, set_initial_colors, find_neighbors, iterations=iterations)

def f_pattern_WL(G, hom_counts=defaultdict(list), verbose=False, iterations=None):
    G = nx.convert_node_labels_to_integers(G)

    def n_set():
        return G, list(G.nodes())

    def set_initial_colors(n):
        return {i: [hashlib.sha224(str(hom_counts[i]).encode('utf-8')).hexdigest()] for i in n}

    def find_neighbors(G, n, node):
        return G.neighbors(node)

    return base_WL(verbose, n_set, set_initial_colors, find_neighbors, iterations=iterations)


def kWL(G, k, verbose=False, iterations=None):
    G = nx.convert_node_labels_to_integers(G)

    def n_set():
        V = list(G.nodes())
        V_k = [comb for comb in itertools.combinations(V, k)]
        return G, V_k
    def set_initial_colors(n):
            return {
                i: [
                    hashlib.sha224(
                        str(
                            sorted(map(lambda e: G.has_edge(*e), itertools.combinations(i, 2)))
                        ).encode('utf-8')).hexdigest()
                ] for i in n
            }
    def find_neighbors(G, V_k, node):
            return [n for n in V_k if len(set(n) - set(V_k[V_k.index(node)])) == 1]
    return base_WL(verbose, n_set, set_initial_colors, find_neighbors, iterations=iterations)

def fkWL(G, k, verbose=False, iterations=None):
    G = nx.convert_node_labels_to_integers(G)

    def n_set():
        V = list(G.nodes())
        V_k = [comb for comb in itertools.product(V, repeat=k)]
        return G, V_k

    def set_initial_colors(n):
        return {
            i: [
                hashlib.sha224(
                    str(
                        sorted(map(lambda e: G.has_edge(*e), itertools.combinations(i, 2)))
                    ).encode('utf-8')).hexdigest()
            ] for i in n
        }

    def find_neighbors(G, V_k, node):
        V = list(G.nodes())
        vals = []
        for i in range(k):
            w = []
            nodes_to_add = [u for u in V if u != V_k[V_k.index(node)][i]]
            for u in nodes_to_add:
                aux = list(V_k[V_k.index(node)])
                aux[i] = u
                w.append(tuple(aux))
            vals.extend(w)
        return vals

    return base_WL(verbose, n_set, set_initial_colors, find_neighbors, iterations=iterations)


def compare_graphs(G1, G2, method='WL', k=2, verbose=False):
    methods = {
        'WL': WL,
        'kWL': kWL,
        'fkWL': fkWL,
        'f_pattern_WL': f_pattern_WL
    }

    # If two graphs have different numbers of nodes they cannot be isomorphic
    if len(G1.nodes()) != len(G2.nodes()):
        if verbose:
            print('Non-Isomorphic by different number of nodes!')
        return False

    c1 = methods[method](G1, k=k, verbose=verbose)
    c2 = methods[method](G2, k=k, verbose=verbose)

    return c1 == c2



if __name__ == '__main__':
    G = nx.Graph()
    G_edge_list = [(1, 2), (2, 3), (1, 3), (4, 5), (5, 6), (4, 6)]
    G.add_edges_from(G_edge_list)

    H = nx.Graph()
    H_edge_list = [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 1)]
    H.add_edges_from(H_edge_list)

    print(compare_graphs(G, H, k=3, verbose=True, method='f_pattern_WL'))