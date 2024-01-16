import pickle
from itertools import combinations, chain
from sklearn.model_selection import train_test_split
from torch_geometric.datasets import TUDataset, Planetoid, GNNBenchmarkDataset
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


# https://mail.python.org/pipermail/scipy-user/2011-May/029521.html

import numpy as np

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
        dataset_raw = Planetoid(f'{os.path.dirname(__file__)}/../data/Planetoid', dataset_name)
    elif dataset_name.startswith('SBM_PATTERN'):
        with open(f'{os.path.dirname(__file__)}/../../LGP-GNN/data/SBMs/PATTERN/'+dataset_name+'.pkl',"rb") as f:
            f = pickle.load(f)
            dataset_raw = f[0]
        return dataset_raw
    else:
        dataset_raw = TUDataset(f'{os.path.dirname(__file__)}/../data/TUDataset', dataset_name, use_node_attr=True)
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

def get_f_pattern_wl_features(dataset, patterns=original_patterns.keys(), iterations=None, use_node_attr=True,
                              count_func=get_graph_hom_counts):
    graph_canonical_forms = []
    colors = []
    for idx in range(len(dataset)):
        graph = to_networkx(dataset[idx])
        counts = count_func(dataset.name, idx, graph.number_of_nodes(), patterns=patterns)
        canonical_form, node_color_his, graph_color_his = f_pattern_WL(graph, hom_counts=counts, verbose=False, iterations=iterations)
        graph_canonical_forms.append(graph_color_his)
        colors.append(get_node_feature_from_colors(dataset[idx].x, node_color_his, use_node_attr=use_node_attr))
    return get_features_from_canonical_forms(len(dataset), graph_canonical_forms), colors

def get_wl_features(dataset, iterations=None, use_node_attr=True):
    graph_canonical_forms = []
    colors = []
    test = set()
    for idx in range(len(dataset)):
        graph = to_networkx(dataset[idx])
        canonical_form, node_color_his, graph_color_his = WL(graph, verbose=False, iterations=iterations)
        test.add(tuple(canonical_form))
        graph_canonical_forms.append(graph_color_his)
        colors.append(get_node_feature_from_colors(dataset[idx].x, node_color_his, use_node_attr=use_node_attr))

    print('# of unique canonical forms: ', len(test))
    return get_features_from_canonical_forms(len(dataset), graph_canonical_forms), colors

def get_kwl_features(dataset, k=3, iterations=None, use_node_attr=True):
    graph_canonical_forms = []
    colors = []
    for idx in range(len(dataset)):
        graph = to_networkx(dataset[idx])
        canonical_form, node_color_his, graph_color_his = kWL(graph, k=k, verbose=False, iterations=iterations)
        graph_canonical_forms.append(graph_color_his)
        colors.append(get_node_feature_from_colors(dataset.x, node_color_his, use_node_attr=use_node_attr))
    return get_features_from_canonical_forms(len(dataset), graph_canonical_forms), colors

def get_fkwl_features(dataset, k=3, iterations=None, use_node_attr=True):
    graph_canonical_forms = []
    colors = []
    for idx in range(len(dataset)):
        graph = to_networkx(dataset[idx])
        canonical_form, node_color_his, graph_color_his = fkWL(graph, k=k, verbose=False, iterations=iterations)
        graph_canonical_forms.append(graph_color_his)
        colors.append(get_node_feature_from_colors(dataset.x, node_color_his, use_node_attr=use_node_attr))
    return get_features_from_canonical_forms(len(dataset), graph_canonical_forms), colors


def get_features_from_canonical_forms(num_graphs, graph_canonical_forms, normalize=True):
    canonical_forms = list(map(lambda c: list(chain(*c)), graph_canonical_forms))
    hist = chain(*canonical_forms)
    color_keys = sorted(list(set(map(lambda c: c[0], hist))))
    features = torch.zeros(num_graphs, len(color_keys))
    for graph, graph_colors in enumerate(canonical_forms):
        for color in graph_colors:
            features[graph, color_keys.index(color[0])] = color[1]
    if normalize:
        # features = torch.log10(features+1)
        # m = torch.mean(features, dim=0)
        # s = torch.std(features, dim=0)
        # features = (features - m) / s
        features = torch.nn.functional.normalize(features, dim=0, p=2)
    return features

def get_node_feature_from_colors(node_attr, colors, normalize=True, use_node_attr=True):
    # colors = [colors[-1]]
    # hist = chain(*(chain(*list(map(lambda a:list(map(lambda b:list(filter(lambda c:c[1]>1, b)), a)), colors)))))
    hist = chain(*(chain(*list(colors))))
    color_keys = sorted(list(set(map(lambda c: c[0], hist))))
    features = torch.zeros(len(colors[0]), len(color_keys))
    for iter in colors:
        for node, node_colors in enumerate(iter):
            for color in node_colors:
                if color[1] > 1:
                    features[node, color_keys.index(color[0])] = color[1]
    if use_node_attr:
        features = torch.cat((node_attr, features), 1)
    if normalize:
        # features = torch.log10(features)
        # # set -inf to -1
        # features.clip_(min=-0.001)
        # m = torch.mean(features, dim=0)
        # s = torch.std(features, dim=0)
        # total_counts = (features - m) / s
        # total_counts.nan_to_num_(0)
        features = torch.nn.functional.normalize(features, dim=0, p=2)
    return features


def get_bound_graph(dataset, get_graph_feature_func, train_idx=None, seed=1):
    bounds = []
    np.random.seed(seed)
    features, _ = get_graph_feature_func()
    delta = 0.01
    lip_margin_constant = 4
    m = 0
    for c in range(dataset.num_classes):
        candidates = train_idx if train_idx is not None else torch.range(len(dataset))
        candidates = candidates[dataset.data.y[candidates] == c]
        sample_size = min(len(candidates), 100)
        c_idx = np.random.choice(candidates, sample_size, replace=False)
        num_samples = int(len(c_idx)/2)
        dkl = KLdivergence(
            features[c_idx[:num_samples]],
            features[c_idx[num_samples:num_samples * 2]],
        )
        dist = torch.cdist(features[c_idx], features[c_idx])
        diameter = dist.max().item()
        dkl_tilde = diameter*math.sqrt(min(dkl/2, 1-torch.exp(-torch.tensor(dkl))))/num_samples
        print(f'dkl: {dkl}, diameter: {diameter}, dkl_tilde: {dkl_tilde}')
        bound = (dkl_tilde +
                 2 * diameter * math.sqrt(
                    math.log(2 * dataset.num_classes / delta, 10) /
                    (num_samples * math.floor(len(candidates) / 2 / num_samples))
                )) * lip_margin_constant
        bounds.append(bound)
        m += math.floor(len(candidates)/2/num_samples)
    bound = sum(bounds)/len(bounds) + math.sqrt(math.log(2/delta)/2/m)
    # print(f'bound: {bound}')
    return bound

def get_bound_node(dataset, get_graph_feature_func, train_idx=None, seed=1):
    bounds = []
    np.random.seed(seed)
    _, node_features = get_graph_feature_func()
    node_features = node_features[0]
    delta = 0.01
    lip_margin_constant = 6
    m = 0
    for c in range(dataset.num_classes):
        candidates = train_idx if train_idx is not None else torch.range(len(dataset))
        candidates = candidates[dataset.data.y[candidates] == c]
        sample_size = min(len(candidates), 300)
        c_idx = np.random.choice(candidates, sample_size, replace=False)
        num_samples = int(len(c_idx)/2)
        dkl = KLdivergence(
            node_features[c_idx[:num_samples]],
            node_features[c_idx[num_samples:num_samples * 2]],
        )
        dist = torch.cdist(node_features[c_idx], node_features[c_idx])
        diameter = dist.max().item()
        dkl_tilde = diameter*math.sqrt(min(dkl/2, 1-torch.exp(-torch.tensor(dkl))))/num_samples
        print(f'dkl: {dkl}, diameter: {diameter}, dkl_tilde: {dkl_tilde}')
        bound = (dkl_tilde +
                 2*diameter*math.sqrt(
                    math.log(2*dataset.num_classes/delta, 10)/
                    (num_samples*math.floor(len(candidates)/2/num_samples))
                ))*lip_margin_constant
        bounds.append(bound)
        m += math.floor(len(candidates)/2/num_samples)
    return sum(bounds)/len(bounds) + math.sqrt(math.log(2/delta)/2/m)

def run_node_tests(dataset_name, iterations=None, seed=1, use_node_attr=True, c_type='HOM'):
    res = []
    dataset = get_dataset(dataset_name)

    print(f'iterations: {iterations}')
    one_wl_bound = get_bound_node(dataset, train_idx=dataset.data.train_mask.nonzero().view(-1), seed=seed,
                                  get_graph_feature_func=lambda: get_wl_features(dataset, iterations=iterations, use_node_attr=use_node_attr))
    print('1-WL:' + str(one_wl_bound))
    res.append(['1-WL', one_wl_bound])
    print('F-pattern WL:')
    # for patterns in combinations(list(original_patterns.keys()),1):
    for patterns in [
        ['2-path','3-path','4-path','5-path'], ['triangle','4-cycle','5-cycle','6-cycle'], ['triangle','4-clique','5-clique','6-clique']
    ]:
        f_wl_bound = get_bound_node(dataset, train_idx=dataset.data.train_mask.nonzero().view(-1), seed=seed,
                                    get_graph_feature_func=lambda: get_f_pattern_wl_features(dataset, patterns=patterns,
                                     iterations=iterations, use_node_attr=use_node_attr,
                                     count_func=get_graph_hom_counts if c_type=='HOM' else get_graph_subgraph_counts))
        print(f'patterns: {patterns}, bound: {f_wl_bound}')
        res.append([patterns, f_wl_bound])
    # print('3-WL:')
    # get_bound_graph(dataset, get_graph_feature_func=lambda : get_kwl_features(dataset, k=3, iterations=iterations))
    # print('3-FWL:')
    # get_bound_graph(dataset, get_graph_feature_func=lambda : get_fkwl_features(dataset, k=3, iterations=iterations))
    return res

def run_pattern_tests(dataset_name, iterations=None, seed=1, use_node_attr=True, c_type='HOM', normalize=True):
    import networkx as nx
    res = []
    dataset = get_dataset(dataset_name)

    np.random.seed(seed)
    delta = 0.01
    lip_margin_constant = 6
    c_idx = np.random.choice(range(len(dataset)), 100, replace=False)
    labels = []
    node_color_his_list = []
    for idx in c_idx:
        graph = nx.Graph(dataset[idx][0].to_networkx())
        if dataset[idx][0].ndata['feat'].shape[1] <= 3:
            canonical_form, node_color_his, graph_color_his = WL(graph, verbose=False, iterations=iterations)
        else:
            counts = dict(zip(list(range(dataset[idx][0].num_nodes())), dataset[idx][0].ndata['feat'][:, 3:].tolist()))
            canonical_form, node_color_his, graph_color_his = f_pattern_WL(graph, hom_counts=counts, verbose=False, iterations=iterations)
        node_color_his_list.append(node_color_his)
        labels += dataset[idx][1]

    max_iterations = max([len(node_color_his) for node_color_his in node_color_his_list])
    colors = [set() for _ in range(max_iterations)]
    for node_color_his in node_color_his_list:
        for it in range(len(node_color_his)):
            for n in node_color_his[it]:
                for c in n:
                    colors[it].add(c[0])

    for node_color_his in node_color_his_list:
        for it in range(len(node_color_his)):
            for n in node_color_his[it]:
                for c in colors[it]:
                    if not any(filter(lambda x: x[0] == c, n)):
                        n.append((c, 0))

    node_features = []
    for i in range(len(c_idx)):
        idx = c_idx[i]
        node_features.append(
            get_node_feature_from_colors(dataset[idx][0].ndata['feat'][:, :3], node_color_his_list[i], use_node_attr=use_node_attr))
    node_features = torch.stack(list(chain(*node_features)))
    labels = torch.stack(labels)

    num_classes = dataset[0][1].max()+1
    m = 0
    bounds = []
    for c in range(num_classes):
        candidates = torch.arange(len(labels))
        candidates = candidates[labels[candidates] == c]
        sample_size = min(len(candidates), 300)
        c_idx = np.random.choice(candidates, sample_size, replace=False)
        num_samples = int(len(c_idx)/2)
        dkl = KLdivergence(
            node_features[c_idx[:num_samples]],
            node_features[c_idx[num_samples:num_samples * 2]],
        )
        dist = torch.cdist(node_features[c_idx], node_features[c_idx])
        diameter = dist.max().item()
        dkl_tilde = diameter*math.sqrt(min(dkl/2, 1-torch.exp(-torch.tensor(dkl))))/num_samples
        print(f'dkl: {dkl}, diameter: {diameter}, dkl_tilde: {dkl_tilde}')
        bound = (dkl_tilde +
                 2*diameter*math.sqrt(
                    math.log(2*num_classes/delta, 10)/
                    (num_samples*math.floor(len(candidates)/2/num_samples))
                ))*lip_margin_constant
        bounds.append(bound)
        m += math.floor(len(candidates)/2/num_samples)
    final_bound = sum(bounds) / len(bounds) + math.sqrt(math.log(2 / delta) / 2 / m)
    print(f'{dataset_name}: {final_bound}')
    res.append((dataset_name, final_bound))
    return res

def run_graph_tests(dataset_name, iterations=None, seed=1, use_node_attr=True, c_type='HOM'):
    res = []
    dataset = get_dataset(dataset_name)
    print(f'iterations: {iterations}')
    one_wl_bound = get_bound_graph(dataset, train_idx=dataset.data.train_graph_index, seed=seed,
                                  get_graph_feature_func=lambda: get_wl_features(dataset, iterations=iterations, use_node_attr=use_node_attr))
    print('1-WL:' + str(one_wl_bound))
    res.append(['1-WL', one_wl_bound])
    print('F-pattern WL:')
    # for patterns in combinations(list(original_patterns.keys()),1):
    # for patterns in [['3-star']]:
    for patterns in [
        ['2-path', '3-path', '4-path', '5-path'], ['triangle', '4-cycle', '5-cycle', '6-cycle'],
        ['triangle', '4-clique', '5-clique', '6-clique']
    ]:
        f_wl_bound = get_bound_graph(dataset, train_idx=dataset.data.train_graph_index, seed=seed,
                                    get_graph_feature_func=lambda: get_f_pattern_wl_features(dataset, patterns=patterns,
                                    iterations=iterations, use_node_attr=use_node_attr,
                                    count_func=get_graph_hom_counts if c_type=='HOM' else get_graph_subgraph_counts))
        print(f'patterns: {patterns}, bound: {f_wl_bound}')
        res.append([patterns, f_wl_bound])
    # print('3-WL:')
    # get_bound_graph(dataset, get_graph_feature_func=lambda : get_kwl_features(dataset, k=3, iterations=iterations))
    # print('3-FWL:')
    # get_bound_graph(dataset, get_graph_feature_func=lambda : get_fkwl_features(dataset, k=3, iterations=iterations))
    return res


if __name__ == '__main__':
    # for i in range(1,6):
    #     run_node_tests('CiteSeer', iterations=i)
    filepath = './fig/test.csv'
    if os.path.exists(filepath):
        os.remove(filepath)
    # iterations = [7]
    iterations = [None]
    with open(filepath, 'a') as f:
        f.writelines('dataset iteration patterns bound bound_std\n')
        for dataset in ['SBM_PATTERN','SBM_PATTERN_345Cl','SBM_PATTERN_34Cl','SBM_PATTERN_5Cl']:
        # for dataset in ['SBM_PATTERN_345Cl','SBM_PATTERN_34Cl','SBM_PATTERN_5Cl']:
        # for dataset in ['ENZYMES','PROTEINS','PTC_MR']:
        # for dataset in ['Cora','CiteSeer','PubMed']:
            print(dataset)
            for iteration in iterations:
                bounds = defaultdict(list)
                for repeat in range(5):
                    res = run_pattern_tests(dataset, iterations=iteration, seed=1+repeat, use_node_attr=True, c_type='HOM')
                    # res = run_graph_tests(dataset, iterations=iteration, seed=1+repeat, use_node_attr=True, c_type='HOM')
                    # res = run_node_tests(dataset, iterations=iteration, seed=1+repeat, use_node_attr=True, c_type='HOM')
                    for r in res:
                        bounds[','.join(r[0])].append(r[1])
                for key in bounds:
                    f.writelines(f'{dataset} {iteration} {key} {np.mean(bounds[key])} {np.std(bounds[key])}\n')
