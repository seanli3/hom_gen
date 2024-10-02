import pickle
from itertools import combinations, chain
from sklearn.model_selection import train_test_split
from torch_geometric.datasets import TUDataset, Planetoid, GNNBenchmarkDataset, MoleculeNet
from torch_geometric.utils import to_networkx
from rooted_hom_count.count_hom import get_hom_count_filename, get_subgraph_count_filename, patterns as original_patterns
import pandas as pd
from collections import defaultdict, OrderedDict
from wl.kwl import f_pattern_WL, WL, kWL, fkWL
import torch
import os
import math
from collections import Counter, defaultdict
from numpy import linalg as LA
import pathlib
from scipy.special import kl_div
import ot
from sklearn.metrics import pairwise_distances

import torch.nn.functional as F


# https://mail.python.org/pipermail/scipy-user/2011-May/029521.html

import numpy as np

def W1_distance(feats_0, feats_1):
    feat_dim = feats_0.shape[-1]
    k = len(feats_0)
    M = pairwise_distances(feats_0, feats_1)
    uni = np.zeros(k) + 1. / k
    kv = ot.emd2(uni, uni, M)
    return kv

def total_variatoin(x,y):
    # x = x / x.sum(dim=1).view(-1,1)
    # y = y / y.sum(dim=1).view(-1,1)
    # return 0.5*(x-y).abs().sum(dim=1).sum()/x.shape[0]
    prod = 1
    for i in range(x.shape[1]):
        max_val = max(x[:,i].max().int().item(), y[:,i].max().int().item())
        x_bins = x[:,i].long().bincount(minlength=int(max_val+1))
        y_bins = y[:,i].long().bincount(minlength=int(max_val+1))
        x_bins = x_bins / x_bins.sum()
        y_bins = y_bins / y_bins.sum()
        prod *=  1 - 0.5*np.abs(x_bins-y_bins).sum()
    return 1 - prod

def KLdivergence(x, y):
  """Compute the Kullback-Leibler divergence between two multivariate samples.
  https://gist.github.com/atabakd/ed0f7581f8510c8587bc2f41a094b518
  Parameters
  ----------
  x : 2D array (n,d)
    Samples from distribution P, which typically represents the true
    distribution.
  y : 2D array (m,d)
    Samples from distribution Q, which typically represents the approximate
    distribution.
  Returns
  -------
  out : float
    The estimated Kullback-Leibler divergence D(P||Q).
  References
  ----------
  Pérez-Cruz, F. Kullback-Leibler divergence estimation of
continuous distributions IEEE International Symposium on Information
Theory, 2008.
  """
  from scipy.spatial import cKDTree as KDTree

  # Check the dimensions are consistent
  x = np.atleast_2d(x)
  y = np.atleast_2d(y)

  n,d = x.shape
  m,dy = y.shape

  assert(d == dy)


  # Build a KD tree representation of the samples and find the nearest neighbour
  # of each point in x.
  xtree = KDTree(x)
  ytree = KDTree(y)

  # Get the first two nearest neighbours for x, since the closest one is the
  # sample itself.
  r = xtree.query(x, k=2, eps=.01, p=2)[0][:,1]
  s = ytree.query(x, k=1, eps=.01, p=2)[0]
  r.clip(min=1e-4, out=r)  # get rid of -inf
  s.clip(min=1e-4, out=s)  # get rid of -inf

  # There is a mistake in the paper. In Eq. 14, the right side misses a negative sign
  # on the first term of the right hand side.
  return np.abs(-np.log(r/s).sum() * d / n + np.log(m / (n - 1.)))

# Assuming perfect classifier
def lipschitz(model, batch):
    x, edge_index = batch.x, batch.edge_index
    for conv in model.convs:
        x = conv(x, edge_index)
        x = torch.nn.functional.relu(x)
    batch.x = x
    output = model.post_mp(batch)[0]
    # output = torch.nn.functional.log_softmax(output, dim=-1)
    embeddings = batch.x

    label = batch.y[batch.train_mask] if 'train_mask' in batch else batch.y
    outputs_1 = []
    outputs_2 = []
    for i in range(len(label)):
        y1 = label[i]
        y2 = np.argmax(np.concatenate([output[i][:label[i]].detach(), np.array([-10000]), output[i][label[i] + 1:].detach()]))
        outputs_1.append(output[i, y1])
        outputs_2.append(output[i, y2])
    outputs_1 = torch.stack(outputs_1)
    outputs_2 = torch.stack(outputs_2)

    grad_1 = torch.autograd.grad(outputs_1, embeddings, grad_outputs=torch.ones(outputs_1.shape[0]), retain_graph=True)[0]
    grad_2 = torch.autograd.grad(outputs_2, embeddings, grad_outputs=torch.ones(outputs_2.shape[0]), retain_graph=True)[0]
    grad_1 = grad_1.numpy().reshape(grad_1.shape[0], -1)
    grad_2 = grad_2.numpy().reshape(grad_2.shape[0], -1)
    grad_norm = LA.norm(grad_1 - grad_2, ord=2, axis=1)

    margin = outputs_1.detach() - outputs_2.detach()

    return np.max(grad_norm), np.median(margin)

class DotDict(dict):
    def __init__(self, **kwds):
        self.update(kwds)
        self.__dict__ = self

def get_dataset(dataset_name, split=[0.6, 0.4]):
    if dataset_name in ['Cora', 'CiteSeer', 'PubMed']:
        dataset_raw = Planetoid(f'{pathlib.Path(__file__).parent.resolve()}/../data/Planetoid', dataset_name)
    elif dataset_name.startswith('SBM_PATTERN'):
        with open(f'{pathlib.Path(__file__).parent.resolve()}/../../LGP-GNN/data/SBMs/'+dataset_name+'.pkl',"rb") as f:
            f = pickle.load(f)
            dataset_raw = f[0]
        return dataset_raw
    elif dataset_name in ["ESOL", "FreeSolv", "Lipo", "PCBA", "MUV", "HIV", "BACE", "BBBP", "Tox21", "ToxCast", "SIDER", "ClinTox"]:
        dataset_raw = MoleculeNet(f'{pathlib.Path(__file__).parent.resolve()}/../data', dataset_name )
        dataset_raw.data.x = dataset_raw.data.x.float()
        dataset_raw.data.y = dataset_raw.data.y.long()
        if dataset_raw.data.y.shape[1] > 1:
            dataset_raw.data.y = dataset_raw.data.y[:,21].view(-1)
        else:
            dataset_raw.data.y = dataset_raw.data.y.view(-1)
    else:
        name = dataset_name.split('TU_')[1]
        dataset_raw = TUDataset(f'{pathlib.Path(__file__).parent.resolve()}/../data/TUDataset', name, use_node_attr=False)
    if len(dataset_raw)>1:
        # graph dataset
        del dataset_raw.data.train_graph_index
        del dataset_raw.data.val_graph_index
        del dataset_raw.data.test_graph_index
        num_graphs = dataset_raw.data.y.shape[0]

        train_index, val_test_index = train_test_split(torch.arange(num_graphs), test_size=1-split[0], random_state=41)
        if len(split) == 2:
            test_index = val_index = val_test_index
        elif len(split) == 3:
            val_index, test_index = train_test_split(val_test_index, test_size=split[2] / (split[1] + split[2]),
                                                     random_state=41)
        else:
            raise ValueError("Invalid split")

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
    return dataset_raw

def get_hom_counts(dataset_name, index, pattern_name):
    filename = get_hom_count_filename(dataset_name, index, pattern_name)
    counts = pd.read_csv(f'{filename}', header=None).values
    return defaultdict(lambda :0, zip(*counts.T))

def get_subgraph_counts(dataset_name, index, pattern_name):
    filename = get_subgraph_count_filename(dataset_name, index, pattern_name)
    counts = pd.read_csv(f'{filename}', header=None).values
    return defaultdict(lambda :0, zip(*counts.T))

def get_graph_hom_counts(dataset_name, index, num_nodes, patterns):
    hom_counts = []
    for pattern_name in patterns:
        counts = get_hom_counts(dataset_name, index, pattern_name)
        hom_counts.append([counts[i] for i in range(num_nodes)])
    # transpose
    hom_counts = list(zip(*hom_counts))
    return OrderedDict(zip(range(len(hom_counts)), hom_counts))

def get_graph_subgraph_counts(dataset_name, index, num_nodes, patterns):
    subgraph_counts = []
    for pattern_name in patterns:
        counts = get_subgraph_counts(dataset_name, index, pattern_name)
        subgraph_counts.append([counts[i] for i in range(num_nodes)])
    # transpose
    subgraph_counts = list(zip(*subgraph_counts))
    return OrderedDict(zip(range(len(subgraph_counts)), subgraph_counts))

def save_f_pattern_wl_features(dataset, patterns=original_patterns.keys(), iterations=None, use_node_attr=False,
                              count_func=get_graph_hom_counts):
    graph_canonical_forms = []
    # node_canonical_forms = []
    for idx in range(len(dataset)):
        graph = to_networkx(dataset[idx])
        counts = count_func(dataset.name, idx, graph.number_of_nodes(), patterns=patterns)
        canonical_form, node_color_his, graph_color_his = f_pattern_WL(
            graph, hom_counts=counts, verbose=False, iterations=iterations,
            node_features=dataset[idx].x.numpy().astype('float16').tolist() if use_node_attr else defaultdict(list))
        graph_canonical_forms.append(graph_color_his)
        # node_canonical_forms.append(node_color_his)

    graph_features = get_features_from_canonical_forms(graph_canonical_forms)
    # node_features = get_node_feature_from_canonical_forms(node_canonical_forms)
    for i in range(iterations):
        torch.save(graph_features[i], f'graph_features/{dataset.name}_wl_features_iter_{i+1}_node_feature={use_node_attr}_patterns={",".join(patterns)}.pt')
        # torch.save(node_features[i], f'node_features/{dataset.name}_wl_features_iter_{i+1}_node_feature={use_node_attr}_patterns={",".join(patterns)}.pt')

def get_pattern_counts(dataset, patterns=original_patterns.keys(), iterations=None, use_node_attr=True,
                              count_func=get_graph_hom_counts):
    graph_canonical_forms = []
    features = []
    unique_graph_colors = set()
    unique_node_colors = set()
    for idx in range(len(dataset)):
        graph = to_networkx(dataset[idx])
        counts = count_func(dataset.name, idx, graph.number_of_nodes(), patterns=patterns)
        features.append(np.sum(np.array([list(v) for v in counts.values()]),0))
    return np.array(features)


def save_wl_features(dataset, iterations=None, use_node_attr=False):
    graph_canonical_forms = []
    # node_canonical_forms = []
    for idx in range(len(dataset)):
        graph = to_networkx(dataset[idx])
        canonical_form, node_color_his, graph_color_his = WL(
            graph, node_features=dataset[idx].x.numpy().astype('float16').tolist() if use_node_attr else defaultdict(list),
            verbose=False, iterations=iterations)
        graph_canonical_forms.append(graph_color_his)
        # node_canonical_forms.append(node_color_his)

    graph_features = get_features_from_canonical_forms(graph_canonical_forms)
    # node_features = get_node_feature_from_canonical_forms(node_canonical_forms)
    for i in range(iterations):
        torch.save(graph_features[i], f'graph_features/{dataset.name}_wl_features_iter_{i+1}_node_feature={use_node_attr}.pt')
        # torch.save(node_features[i].to_sparse_coo(), f'node_features/{dataset.name}_wl_features_iter_{i+1}_node_feature={use_node_attr}.pt')

def save_kwl_features(dataset, k=3, iterations=None, use_node_attr=False):
    graph_canonical_forms = []
    for idx in range(len(dataset)):
        graph = to_networkx(dataset[idx])
        canonical_form, node_color_his, graph_color_his = kWL(graph, k=k, verbose=False, iterations=iterations)
        graph_canonical_forms.append(graph_color_his)

    graph_features = get_features_from_canonical_forms(graph_canonical_forms)
    for i in range(iterations):
        torch.save(graph_features[i], f'features/{dataset.name}_{k}wl_features_iter_{i+1}.pt')

def save_fkwl_features(dataset, k=3, iterations=None, use_node_attr=False):
    graph_canonical_forms = []
    colors = []
    for idx in range(len(dataset)):
        graph = to_networkx(dataset[idx])
        canonical_form, node_color_his, graph_color_his = fkWL(graph, k=k, verbose=False, iterations=iterations)
        graph_canonical_forms.append(graph_color_his)
    graph_features = get_features_from_canonical_forms(graph_canonical_forms)
    for i in range(iterations):
        torch.save(graph_features[i], f'features/{dataset.name}_{k}fwl_features_iter_{i+1}.pt')


def get_features_from_canonical_forms(graph_canonical_forms, normalize=False):
    output = []
    num_graphs = len(graph_canonical_forms)
    for iteration in range(len(graph_canonical_forms[0])):
        canonical_forms = list(map(lambda c: c[iteration], graph_canonical_forms))
        hist = chain(*canonical_forms)
        color_keys = sorted(list(set(map(lambda c: c[0], hist))))
        feature_indices = []
        feature_values = []
        for graph, graph_colors in enumerate(canonical_forms):
            for color in graph_colors:
                feature_indices.append([graph, color_keys.index(color[0])])
                feature_values.append(color[1])
        features = torch.sparse_coo_tensor(torch.tensor(feature_indices).T, feature_values, (num_graphs, len(color_keys)))
        # features = torch.zeros(num_graphs, len(color_keys))
        # for graph, graph_colors in enumerate(canonical_forms):
        #     for color in graph_colors:
        #         features[graph, color_keys.index(color[0])] = color[1]
        # if normalize:
        #     # features = torch.log10(features+1)
        #     # m = torch.mean(features, dim=0)
        #     # s = torch.std(features, dim=0)
        #     # features = (features - m) / s
        #     features = torch.nn.functional.normalize(features, dim=0, p=1)
        output.append(features)
    return output

def get_node_feature_from_canonical_forms(node_canonical_forms, normalize=False):
    output = []
    num_nodes = sum(map(lambda f:len(f[0]), node_canonical_forms))
    for iteration in range(len(node_canonical_forms[0])):
        canonical_forms = list(chain(*map(lambda c: list(c[iteration].values()), node_canonical_forms)))
        hist = chain(*canonical_forms)
        color_keys = sorted(list(set(map(lambda c: c[0], hist))))
        features = torch.zeros(num_nodes, len(color_keys))
        for node, node_colors in enumerate(canonical_forms):
            for color in node_colors:
                features[node, color_keys.index(color[0])] = color[1]
        if normalize:
            # features = torch.log10(features+1)
            # m = torch.mean(features, dim=0)
            # s = torch.std(features, dim=0)
            # features = (features - m) / s
            features = torch.nn.functional.normalize(features, dim=0, p=1)
        output.append(features)
    return output

def compute_graph_bound(dataset, lambda_features, phi_features, gamma=1, use_node_feature=False):
    seed = 41
    num_classes = dataset.num_classes
    y = dataset.data.y
    num_graphs = len(y)
    if dataset.name == 'MCF-7':
        import numpy as np
        np.random.seed(41)
        train_idx = np.random.choice((dataset.data.y == 0).nonzero().view(-1), 1000, replace=False)
        train_idx = np.concatenate(
            [train_idx, np.random.choice((dataset.data.y == 1).nonzero().view(-1), 1000, replace=False)])
    else:
        train_idx, _ = train_test_split(torch.arange(num_graphs), test_size=0.1, random_state=seed)

    lip_f = 0
    # estimate lip_f in batches of 100 samples (so we don't run out of memory on large datasets)
    for i in range(0, len(train_idx), 100):
        if i + 100 > len(train_idx):
            sub_lambda_features_1 = torch.index_select(lambda_features, 0, train_idx[i:len(train_idx)]).to_dense()
            sub_phi_features_1 = torch.index_select(phi_features, 0, train_idx[i:len(train_idx)])
        else:
            sub_lambda_features_1 = torch.index_select(lambda_features, 0,
                                                       torch.tensor(train_idx[i:i + 100])).to_dense()
            sub_phi_features_1 = torch.index_select(phi_features, 0, torch.tensor(train_idx[i:i + 100]))
        for j in range(0, len(train_idx), 100):
            if i + 100 > len(train_idx):
                sub_lambda_features_2 = torch.index_select(lambda_features, 0, train_idx[j:len(train_idx)]).to_dense()
                sub_phi_features_2 = torch.index_select(phi_features, 0, train_idx[j:len(train_idx)])
            else:
                sub_lambda_features_2 = torch.index_select(lambda_features, 0,
                                                           torch.tensor(train_idx[j:j + 100])).to_dense()
                sub_phi_features_2 = torch.index_select(phi_features, 0, torch.tensor(train_idx[j:j + 100]))

            lambda_dist = torch.cdist(sub_lambda_features_1, sub_lambda_features_2, p=2)
            phi_dist = torch.cdist(sub_phi_features_1, sub_phi_features_2, p=2)
            div = phi_dist / lambda_dist
            div[div.isnan()] = -9e-9
            div[div.isinf()] = -9e-9
            lip_f = max(div.max(), lip_f)

    # lambda_dist = torch.cdist(lambda_features[train_idx], lambda_features[train_idx], p=2)
    # phi_dist = torch.cdist(phi_features[train_idx], phi_features[train_idx], p=2)
    # min_lambda_dist = lambda_dist.to_sparse().values().min()
    # max_lambda_dist = lambda_dist.to_sparse().values().max()
    # max_phi_dist = phi_dist.max()

    # lip_f = 2
    # lip_f = max_phi_dist / min_lambda_dist
    # lip_f = max_phi_dist / 1
    # div = phi_dist / lambda_dist
    # div[div.isnan()] = -9e-9
    # div[div.isinf()] = -9e-9
    # lip_f = div.max()

    # print(f'lip_f:{lip_f}, min_lambda_dist: {min_lambda_dist}, max_lambda_dist: {max_lambda_dist},  max_phi_dist: {max_phi_dist}')

    # lip_rho = 1
    lip_rho = math.sqrt(num_classes-1)/num_classes

    bounds = []
    np.random.seed(seed)
    delta = 0.1
    m = 0
    n = 1
    wass= []
    diam_cs = []
    for c in range(num_classes):
        candidates = train_idx[y[train_idx] == c]
        m_c = len(candidates)
        avg_wass_dist = 0
        for _ in range(n):
            num_samples = int(len(candidates) / (2*n))
            c_idx_1 = np.random.choice(candidates, num_samples, replace=False)
            c_idx_2 = np.random.choice(candidates, num_samples, replace=False)
            x1 = torch.index_select(lambda_features, 0, torch.tensor(c_idx_1)).to_dense()
            x2 = torch.index_select(lambda_features, 0, torch.tensor(c_idx_2)).to_dense()
            wass_dist = W1_distance(x1, x2)
            avg_wass_dist += wass_dist
        avg_wass_dist /= n
        wass.append(avg_wass_dist)
        # print(f'avg wass dist: {avg_wass_dist}')

        # diameter in Euclidean distance
        diam_c = torch.cdist(phi_features[candidates], phi_features[candidates], p=2).max()
        diam_cs.append(diam_c.detach().item())
        # print(f'diam_c: {diam_c}')

        bound = avg_wass_dist + 2*diam_c*math.sqrt(
            math.log(2 * num_classes / delta, 10) /
            (n * math.floor(m_c / 2 / n))
        )

        bound *= lip_f*lip_rho/gamma

        bounds.append(bound.detach().item())

        m += math.floor(m_c / 2 / n)

    bound = sum(bounds) / len(bounds) + math.sqrt(math.log(2 / delta) / 2 / m)
    # print(f'bound: {bound}')
    # print(f'wass_dist: {np.mean(wass)}, diam_cs: {np.mean(diam_cs)}')
    return bound, lip_f, np.mean(wass), np.std(wass)


def get_bound_graph(dataset, phi_diamters, get_graph_feature_func, train_idx=None, seed=1):
    np.random.seed(seed)
    features, _ = get_graph_feature_func()
    return compute_graph_bound(dataset.num_classes, features, dataset.data.y, phi_diamters, train_idx=train_idx, seed=seed)

def compute_node_bound(num_classes, node_features, y, train_idx=None, seed=1, lip_margin_constant=6):
    np.random.seed(seed)
    delta = 0.01
    m = 0
    bounds = []
    for c in range(num_classes):
        candidates = train_idx if train_idx is not None else torch.range(len(y))
        candidates = candidates[y[candidates] == c]
        sample_size = min(len(candidates), 300)
        c_idx = np.random.choice(candidates, sample_size, replace=False)
        num_samples = int(len(c_idx)/2)
        if len(node_features.shape) > 1:
            dtv = total_variatoin(
                node_features[c_idx[:num_samples]],
                node_features[c_idx[num_samples:num_samples * 2]],
            )
        else:
            min_val = node_features.min()
            # scale to positive numbers
            node_features -= min_val + 1e-5
            dtv = sum(kl_div(node_features[c_idx[:num_samples]], node_features[c_idx[num_samples:num_samples * 2]]))
            node_features = node_features.view(-1,1)
        dist = torch.cdist(node_features[c_idx], node_features[c_idx])
        diameter = dist.max().item()
        dkl_tilde = diameter*math.sqrt(min(dtv/2, 1-torch.exp(-torch.tensor(dtv))))/num_samples
        print(f'dtv: {dtv}, diameter: {diameter}, dkl_tilde: {dkl_tilde}')
        bound = (dkl_tilde +
                 2*diameter*math.sqrt(
                    math.log(2*num_classes/delta, 10)/
                    (num_samples*math.floor(len(candidates)/2/num_samples))
                ))*lip_margin_constant
        bounds.append(bound)
        m += math.floor(len(candidates)/2/num_samples)
    return sum(bounds)/len(bounds) + math.sqrt(math.log(2/delta)/2/m)

def get_bound_node(dataset, get_graph_feature_func, train_idx=None, seed=1):
    _, node_features = get_graph_feature_func()
    np.random.seed(seed)
    node_features = node_features[0]
    return compute_node_bound(dataset.num_classes, node_features, dataset.data.y, train_idx=train_idx, seed=seed)


def get_phi_features(dataset_name, num_layers, repeat, normalize=False, use_node_feature=False, pattern=None):
    class dotdict(dict):
        """dot.notation access to dictionary attributes"""
        __getattr__ = dict.get
        __setattr__ = dict.__setitem__
        __delattr__ = dict.__delitem__

    import run.custom_graphgym  # noqa, register custom modules
    from torch_geometric.graphgym.loader import create_dataset
    from torch_geometric.graphgym.loader import get_loader
    from torch_geometric.graphgym.config import set_cfg
    from torch_geometric.graphgym.config import (
        cfg,
        load_cfg,
    )
    from run.model_builder import create_model

    set_cfg(cfg)
    args = dotdict()
    args.cfg_file = '../run/configs/graph.yaml'
    args.opts = []
    load_cfg(cfg, args)
    cfg.dataset.name = dataset_name
    cfg.dataset.use_node_features = use_node_feature
    cfg.gnn.layers_mp = num_layers
    cfg.model.normalise_embedding = normalize
    if pattern is not None:
        cfg.dataset.patterns = pattern
        cfg.dataset.add_counts = True
    dataset = create_dataset()

    model = create_model(to_device=False, dim_in=dataset.num_features, dim_out=dataset.num_classes)
    loader = get_loader(dataset, cfg.train.sampler, len(dataset), shuffle=False)
    if pattern is None:
        # file_path = f'models/graph-dataset={dataset_name}-node_feat={use_node_feature}-type=GCN-norm={normalize}-l_mp={num_layers}-max_epoch={200}/{repeat}/ckpt/'
        if use_node_feature:
            file_path = f'run/results/graph_grid_gcn/graph-dataset={dataset_name}-node_feat={use_node_feature}-type=GCN-norm={normalize}-l_mp={num_layers}/{repeat}/ckpt/'
    else:
        if use_node_feature:
            file_path = f'models/graph-dataset={dataset_name}-model=GCN-c_type=HOM-l_mp={num_layers}-norm={normalize}-add_counts=True-patterns={pattern}-node_feat={True}/{repeat}/ckpt/'
            # file_path = f'models/graph-dataset={dataset_name}-model=GCN-c_type=HOM-l_mp={num_layers}-norm={normalize}-add_counts=True-patterns={pattern}/{repeat}/ckpt/'
        else:
            # file_path = f'models/graph-dataset={dataset_name}-model=GCN-c_type=HOM-l_mp={num_layers}-norm={normalize}-add_counts=True-patterns={pattern}-node_feat={False}-max_epoch={200}/{repeat}/ckpt/'
            file_path = f'models/graph-dataset={dataset_name}-model=GCN-c_type=HOM-l_mp={num_layers}-node_feat=False-norm={normalize}-add_counts=True-patterns={pattern}/{repeat}/ckpt/'
    file_name = os.listdir(file_path)[0]
    # file_name = 'last.ckpt'
    ckpt = torch.load(file_path + file_name, map_location='cpu')
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    # get embeddings for training set
    pred, _ = model(list(loader)[0])
    return pred


def save_lambda_features(use_node_attr=False):
    iterations = 2
    # for dataset_name in ['TU_ENZYMES']:
    for dataset_name in ['TU_PROTEINS']:
        print(dataset_name)
        dataset = get_dataset(dataset_name)
        save_wl_features(dataset, iterations=iterations, use_node_attr=use_node_attr)
        # save_kwl_features(dataset, k=3, iterations=iterations, use_node_attr=use_node_attr)
        for patterns in [['6-cycle','6-clique']]:
        # for patterns in [['2-path', '3-path', '4-path', '5-path'], ['triangle', '4-clique', '5-clique', '6-clique'], ['triangle', '4-cycle', '5-cycle', '6-cycle']]:
            save_f_pattern_wl_features(dataset, iterations=iterations, use_node_attr=use_node_attr, count_func=get_graph_hom_counts, patterns=patterns)
    print('finished')

def print_bound():
    phi_normalize = False
    use_node_attr = True
    lambda_normalize = False
    patterns = ['2-path,3-path,4-path,5-path', 'triangle,4-cycle,5-cycle,6-cycle',
                'triangle,4-clique,5-clique,6-clique']
    # patterns= ['2-path,3-path,4-path,5-path']
    patterns= [None]
    # patterns= ['triangle,4-cycle,5-cycle,6-cycle']
    # for pattern in [['2-path', '3-path', '4-path', '5-path'], ['triangle', '4-clique', '5-clique', '6-clique'], ['triangle', '4-cycle', '5-cycle', '6-cycle']]:
    # for dataset_name in ['TU_PROTEINS','TU_ENZYMES','TU_MUTAG','BACE','SIDER']:
    for dataset_name in ['TU_PROTEINS']:
        for pattern in patterns:
            print(pattern)
            # for dataset_name in ['SIDER']:
            # for dataset_name in ['SIDER']:
            # for dataset_name in ['SIDER','BACE']:
            #     print(dataset_name)
            dataset = get_dataset(dataset_name)
            for layer in range(1,7):
                # print(f'dataset: {dataset_name}, layer: {layer}')
                if pattern is None:
                    lambda_features = torch.load(f'graph_features/{dataset.name}_wl_features_iter_{layer}_node_feature={use_node_attr}.pt').float()
                else:
                    lambda_features = torch.load(f'graph_features/{dataset.name}_wl_features_iter_{layer}_node_feature={use_node_attr}_patterns={pattern}.pt').float()
                    # lambda_features = torch.load(f'graph_features/{dataset.name}_wl_features_iter_{layer}_node_feature={use_node_attr}_patterns={pattern}.pt').to_dense()
                if lambda_normalize:
                    lambda_features = torch.nn.functional.normalize(lambda_features, dim=1, p=1)
                bounds, lip_fs = [], []
                for repeat in range(5):
                    phi_features = get_phi_features(dataset_name, layer, repeat, normalize=phi_normalize, use_node_feature=use_node_attr, pattern=pattern)
                    bound, lif_f, wass, wass_std = compute_graph_bound(dataset, lambda_features, phi_features, gamma=1, use_node_feature=use_node_attr)
                    bounds.append(bound)
                    lip_fs.append(lif_f.detach())
                # print(f'dataset: {dataset_name}, layer: {layer}, bound: {sum(bounds)/len(bounds)}, bound_std: {np.std(bounds)}')
                print(f'{dataset_name},{layer},{sum(bounds)/len(bounds)},{np.std(bounds)},{np.mean(lip_fs)},{np.std(lip_fs)},{wass},{wass_std}')

if __name__ == '__main__':
    save_lambda_features(use_node_attr=True)
    print_bound()

