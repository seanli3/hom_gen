from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_loader
import networkx as nx
import torch
import pickle
from torch_geometric.utils import degree
from torch_geometric.data import InMemoryDataset
from torch_geometric.datasets import (PPI, Amazon, Coauthor, KarateClub,
                                      MNISTSuperpixels, Planetoid, QM7b, MoleculeNet,
                                      TUDataset, LINKXDataset, WebKB, WikipediaNetwork, Actor, ZINC, Reddit2)
from torch_geometric.graphgym.loader import load_pyg, load_ogb
from sklearn.model_selection import train_test_split
from rooted_hom_count.count_hom import patterns as original_patterns
from wl.compute_bound import get_graph_hom_counts, get_graph_subgraph_counts
from ..synthetic_dataset import SyntheticCycles


def max_degree(graph, k):
    return map(lambda x: x[0], sorted(graph.degree, key=lambda x: x[1], reverse=True)[:k])

def max_reaching_centrality(graph, k):
    q = []
    for v in graph:
        centrality = nx.local_reaching_centrality(graph, v, nx.shortest_path(graph, v))
        if len(q) < k:
            q.append((v,centrality))
        else:
            if centrality > q[0][1]:
                q.pop(0)
                q.append((v,centrality))
            else:
                continue
        q = sorted(q, key=lambda x:x[1])
    return list(map(lambda x:x[0], q))

class NormalizedDegree(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        deg = degree(data.edge_index[0], dtype=torch.float)
        deg = (deg - self.mean) / self.std
        data.x = deg.view(-1, 1)
        return

def load_dataset(format, name, dataset_dir):
    dataset_dir = '{}/{}'.format(dataset_dir, name)
    if format == 'PyG':
        if name in ["penn94", "reed98", "amherst41", "cornell5", "johnshopkins55", "genius"]:
            dataset_raw = LINKXDataset(dataset_dir, name=name)
        elif name in ["Cornell", 'Texas', 'Wisconsin']:
            dataset_raw = WebKB(dataset_dir, name=name)
        elif name in ['Chameleon', 'Squirrel']:
            dataset_raw = WikipediaNetwork(dataset_dir, name=name, geom_gcn_preprocess=True)
        elif name in ["Actor"]:
            dataset_raw = Actor(dataset_dir)
        elif name in ["Reddit2"]:
            dataset_raw = Reddit2(dataset_dir)
        elif name in ['ZINC']:
            train = ZINC(dataset_dir, subset=True, split="train")
            val = ZINC(dataset_dir, subset=True, split="val")
            test = ZINC(dataset_dir, subset=True, split="test")
            dataset_raw = InMemoryDataset()
            train_data = [data for data in train]
            val_data = [data for data in val]
            test_data = [data for data in test]
            dataset_raw.data, dataset_raw.slices = dataset_raw.collate(train_data + val_data + test_data)
            dataset_raw.data.x = torch.nn.functional.one_hot(dataset_raw.data.x.view(-1)).float()
            dataset_raw.data.train_graph_index = torch.arange(len(train))
            dataset_raw.data.val_graph_index = torch.arange(len(train), len(train)+len(val))
            dataset_raw.data.test_graph_index = torch.arange(len(train)+len(val), len(train)+len(val)+len(test))
        elif name[:3] == 'TU_':
            dataset_raw = TUDataset(dataset_dir, name[3:], use_node_attr=True)
            # TU_IMDB doesn't have node features
            if dataset_raw.data.x is None:
                dataset_raw.data.x = torch.ones((dataset_raw.data.num_nodes, 1)).float()
                data_list = [dataset_raw.get(i) for i in dataset_raw.indices()]
                for data in data_list:
                    data.x = torch.ones((data.num_nodes, 1)).float()
                dataset_raw.data, dataset_raw.slices = dataset_raw.collate(data_list)
                del dataset_raw._data_list
        elif name in ["ESOL", "FreeSolv", "Lipo", "PCBA", "MUV", "HIV", "BACE", "BBBP", "Tox21", "ToxCast", "SIDER", "ClinTox"]:
            dataset_raw = MoleculeNet(dataset_dir, name )
            dataset_raw.data.x = dataset_raw.data.x.float()
            dataset_raw.data.y = dataset_raw.data.y.long()
            if dataset_raw.data.y.shape[1] > 1:
                dataset_raw.data.y = dataset_raw.data.y[:,21].view(-1)
            else:
                dataset_raw.data.y = dataset_raw.data.y.view(-1)
        elif name == 'synthetic_cycles':
            dataset_raw = SyntheticCycles(dataset_dir, name)
        else:
            dataset_raw = load_pyg(name, dataset_dir)
    elif format == 'OGB':
        dataset_raw = load_ogb(name.replace('_', '-'), dataset_dir)
        dataset_raw.data.x = dataset_raw.data.x.float()
    elif format == 'nx':
        try:
            with open('{}/{}.pkl'.format(dataset_dir, name), 'rb') as file:
                graphs = pickle.load(file)
        except Exception:
            graphs = nx.read_gpickle('{}/{}.gpickle'.format(dataset_dir, name))
            if not isinstance(graphs, list):
                graphs = [graphs]
        return graphs
    else:
        raise ValueError('Unknown data format: {}'.format(format))

    if not cfg.dataset.use_node_features:
        dataset_raw.data.x = torch.ones((dataset_raw.data.num_nodes, 1)).float()

    add_splits(dataset_raw)
    if cfg.dataset.add_counts:
        add_counts(dataset_raw)

    return dataset_raw

def add_counts(dataset_raw):
    patterns = cfg.dataset.patterns.split(',')
    if len(list(filter(lambda p:p not in original_patterns, patterns))) > 0:
        raise Exception('Invalid patterns, has to be in {}'.format(original_patterns.keys()))

    total_counts = []
    if cfg.dataset.count_type == 'HOM':
        for idx in range(len(dataset_raw)):
            num_nodes = dataset_raw[idx].num_nodes
            counts = get_graph_hom_counts(dataset_raw.name, idx, num_nodes, patterns)
            total_counts += list(counts.values())
    elif cfg.dataset.count_type == 'ISO':
        for idx in range(len(dataset_raw)):
            num_nodes = dataset_raw[idx].num_nodes
            counts = get_graph_subgraph_counts(dataset_raw.name, idx, num_nodes, patterns)
            if sum(list(map(lambda c: counts[c][0], counts))) < 1:
                print(idx)

            total_counts += list(counts.values())
    else:
        raise Exception('Invalid count type')
    total_counts = torch.FloatTensor(total_counts)
    # total_counts = torch.log10(total_counts)
    # # set -inf to -1
    # total_counts.clip_(min=-1)
    # m = torch.mean(total_counts, dim=0)
    # s = torch.std(total_counts, dim=0)
    # total_counts = (total_counts - m) / s
    # total_counts.nan_to_num_(-1)
    dataset_raw.data.x = torch.cat((dataset_raw.data.x, total_counts), 1)
    # dataset_raw.data.x = total_counts
    del dataset_raw._data_list

def add_splits(dataset_raw):
    split = cfg.dataset.split

    if cfg.dataset.task == 'graph':
        del dataset_raw.data.train_graph_index
        del dataset_raw.data.val_graph_index
        del dataset_raw.data.test_graph_index

        num_graphs = dataset_raw.data.y.shape[0]

        if dataset_raw.name != 'MCF-7':
            train_index, val_test_index = train_test_split(torch.arange(num_graphs), test_size=1-split[0], random_state=41)
            if len(split) == 2:
                test_index = val_index = val_test_index
            elif len(split) == 3:
                val_index, test_index = train_test_split(val_test_index, test_size=split[2] / (split[1] + split[2]),
                                                         random_state=41)
            else:
                raise ValueError("Invalid split")
        else:
            import numpy as np
            np.random.seed(41)
            train_index = np.random.choice((dataset_raw.data.y == 0).nonzero().view(-1), 1000, replace=False)
            train_index = np.concatenate([train_index, np.random.choice((dataset_raw.data.y == 1).nonzero().view(-1), 1000, replace=False)])
            val_index = np.arange(num_graphs)
            val_index = np.delete(val_index, train_index)

        dataset_raw.data.train_graph_index = torch.LongTensor(train_index)
        dataset_raw.data.val_graph_index = torch.LongTensor(val_index)
        dataset_raw.data.test_graph_index = torch.LongTensor(val_index)
    else:
        del dataset_raw.data.train_mask
        del dataset_raw.data.val_mask
        del dataset_raw.data.test_mask

        num_nodes = dataset_raw.data.num_nodes

        train_index, val_test_index = train_test_split(torch.arange(num_nodes), test_size=1-split[0], random_state=41)
        if len(split) == 2:
            test_index = val_index = val_test_index
        elif len(split) == 3:
            val_index, test_index = train_test_split(val_test_index, test_size=split[2]/(split[1]+split[2]), random_state=41)
        else:
            raise ValueError("Invalid split")
        train_mask = torch.zeros(dataset_raw.data.num_nodes).bool()
        train_mask.scatter_(0, train_index, True)
        val_mask = torch.zeros(dataset_raw.data.num_nodes).bool()
        val_mask.scatter_(0, val_index, True)
        dataset_raw.data.train_mask = train_mask
        dataset_raw.data.val_mask = val_mask
        test_mask = torch.zeros(dataset_raw.data.num_nodes).bool()
        test_mask.scatter_(0, test_index, True)
        dataset_raw.data.test_mask = test_mask


register_loader('hom_gen_loader', load_dataset)
