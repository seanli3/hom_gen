import os.path as osp
import random
from typing import Callable, Optional
import networkx as nx
import numpy as np

import torch

from torch_geometric.data import InMemoryDataset, download_url, extract_gz


class SyntheticCycles(InMemoryDataset):
    name = 'synthetic_cycles'

    def __init__(
        self,
        root: str,
        name: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
    ):
        self.name = name.lower()
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self) -> str:
        return f'{self.name}.csv'

    @property
    def processed_file_names(self) -> str:
        return 'synthetic.pt'

    def process(self):
        from torch_geometric.data import Data
        random.seed(1)
        np.random.seed(1)
        data_list = []
        for _ in range(500):
            num_nodes_a = random.randint(2, 10)
            graph_a = nx.generators.random_tree(num_nodes_a)
            data = Data(x=torch.ones(num_nodes_a, 1).float(), y=torch.tensor([0]), edge_index=torch.tensor(list(graph_a.edges)).T, edge_attr=None)
            assert data.edge_index.max() < data.num_nodes
            data_list.append(data)
        for _ in range(500):
            num_nodes_b = random.randint(2, 10)
            graph_b = nx.generators.random_tree(num_nodes_b)
            cycle_nodes = random.randint(3,6)
            cycle = nx.cycle_graph(cycle_nodes)
            graph = nx.disjoint_union(graph_b, cycle)
            graph = nx.contracted_nodes(graph, num_nodes_b-1, num_nodes_b)
            # recreate graph to avoid empty nodes
            graph = nx.from_scipy_sparse_array(nx.adjacency_matrix(graph))
            data = Data(x=torch.ones(graph.number_of_nodes(), 1).float(), y=torch.tensor([1]), edge_index=torch.tensor(list(graph.edges)).T, edge_attr=None)
            assert data.edge_index.max() < data.num_nodes
            data_list.append(data)

        torch.save(self.collate(data_list), self.processed_paths[0])

    def __repr__(self) -> str:
        return f'{self.name}({len(self)})'
