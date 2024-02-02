import homlib as hl
import numpy as np
from homlib import Graph as hlGraph
import networkx as nx
from torch_geometric.datasets import TUDataset, Planetoid
from collections import OrderedDict
import os
from torch_geometric.utils import to_networkx

patterns = OrderedDict({
    # 2 nodes
    '1-path': nx.path_graph(2),
    # 3 nodes
    '2-path': nx.path_graph(3),
    'triangle': nx.cycle_graph(3),
    # 4 nodes
    '3-path': nx.path_graph(4),
    '4-cycle': nx.cycle_graph(4),
    '4-clique': nx.complete_graph(4),
    # 5 nodes
    '4-path': nx.path_graph(5),
    '5-cycle': nx.cycle_graph(5),
    '5-clique': nx.complete_graph(5),
    # 6 nodes
    '5-path': nx.path_graph(6),
    '6-clique': nx.complete_graph(6),
    '6-cycle': nx.cycle_graph(6),
})


def hom_tree(F, G):
    """Specialized tree homomorphism in Python (serializable).
    Add `indexed` parameter to count for each index individually.
    """
    def rec(x, p):
        hom_x = np.ones(G.number_of_nodes(), dtype=float)
        for y in F.neighbors(x):
            if y == p:
                continue
            hom_y = rec(y, x)
            aux = [np.sum(hom_y[list(G.neighbors(a))]) for a in G.nodes()]
            hom_x *= np.array(aux)
        return hom_x
    hom_r = rec(0, -1)
    return np.sum(hom_r)


def nx2homg(nxg):
    """Convert nx graph to homlib graph format. Only
    undirected graphs are supported.
    Note: This function expects nxg to have consecutive integer index."""
    n = nxg.number_of_nodes()
    G = hlGraph(n)
    for (u, v) in nxg.edges():
        G.addEdge(u,v)
    return G


def hom(F, G, use_py=False, density=False):
    """Wrapper for the `hom` function in `homlib`."""
    # Default homomorphism function
    hom_func = hl.hom
    # Check if tree, then change the hom function
    if use_py:
        hom_func = hom_tree
    # Check and convert graph type
    if density:
        scaler = 1.0 / (G.number_of_nodes() ** F.number_of_nodes())
    else:
        scaler = 1.0
    if not use_py:
        F = nx2homg(F)
        G = nx2homg(G)
    return hom_func(F, G) * scaler

def get_hom_count_filename(dataset_name):
    return f'{os.path.dirname(os.path.abspath(__file__))}/tmp/hom_counts/{dataset_name}.csv'


if __name__ == '__main__':
    graph_dataset_names = [
        # 'MUTAG',
        'PTC_MR',
        # 'NCI1',
        'PROTEINS',
        # 'IMDB-BINARY',
        # 'IMDB-MULTI',
        # 'REDDIT-BINARY',
        # 'REDDIT-MULTI-5K',
        # 'COLLAB',
        # 'DD',
        'ENZYMES']
    node_dataset_names = ['Cora', 'CiteSeer', 'PubMed']

    os.makedirs(f'./tmp/hom_counts', exist_ok=True)

    for dataset_name in graph_dataset_names:
        dataset = TUDataset('../data/TUDataset', dataset_name, use_node_attr=True)
        hom_file_name = get_hom_count_filename(dataset_name)
        with open(hom_file_name, 'w') as f:
            for pattern_name in patterns:
                pattern = patterns[pattern_name]
                for index, graph in enumerate(dataset):
                    try:
                        if index == 87:
                            print()
                        hom_count = hom(pattern, to_networkx(graph).to_undirected(), use_py=False, density=False)
                        line = f'{dataset_name} {index} {pattern_name} {hom_count}\n'
                        print(line)
                        f.writelines(line)
                    except Exception as e:
                        print(f'{index}: Error: {e}')

    for dataset_name in node_dataset_names:
        dataset = Planetoid('../data/Planetoid', dataset_name)
        hom_file_name = get_hom_count_filename(dataset_name)
        for pattern_name in patterns:
            pattern = patterns[pattern_name]
            with open(hom_file_name, 'w') as f:
                for index, graph in enumerate(dataset):
                    hom_count = hom(pattern, to_networkx(graph).to_undirected(), use_py=False, density=False)
                    line = f'{dataset_name} {index} {pattern_name} {hom_count}\n'
                    print(line)
                    f.writelines(line)
