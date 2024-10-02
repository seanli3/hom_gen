from collections import OrderedDict
from torch_geometric.datasets import TUDataset, Planetoid, MoleculeNet
import os
import subprocess
# from run.custom_graphgym.synthetic_dataset import SyntheticCycles

patterns = OrderedDict({
    # 2 nodes
    # '1-path': 'A-B;',
    # 3 nodes
    # '2-path': 'A-B;B-C;',
    'triangle': "A-B;B-C;C-A;",
    # '2-star': 'A-B;A-C;',
    # 4 nodes
    # '3-path': 'A-B;B-C;C-D;',
    # '3-star': 'A-B;A-C;A-D;',
    '4-cycle': 'A-B;B-C;C-D;D-A;',
    # 'trailing-triangle': "A-B;B-D;D-A;C-D;",
    # 'chordal-square': "A-B;B-D;D-A;B-C;C-D;",
    '4-clique': "A-B;B-C;C-D;D-A;A-C;B-D;",
    # 5 nodes
    # '4-path': 'A-B;B-C;C-D;D-E;',
    # '4-star': 'A-B;A-C;A-D;A-E;',
    '5-cycle': 'A-B;B-C;C-D;D-E;E-A;',
    # 'house': "A-B;B-C;C-D;D-E;E-A;B-E;",
    # '3-triangle': "A-B;B-C;C-D;D-E;E-A;B-E;C-E;",
    # 'solar-square': "A-B;B-C;C-D;D-E;E-A;A-D;B-E;C-E;",
    # 'near-5-clique': "A-B;B-C;C-D;D-E;E-A;A-C;A-D;B-D;B-E;C-E;",
    '5-clique': "A-B;B-C;C-D;D-E;E-A;A-C;A-D;B-D;B-E;C-E;",
    # '5-clique-minus-one': "A-B;B-C;C-D;D-E;E-A;A-C;B-D;B-E;C-E;",
    # 'lolipop': "A-B;B-C;C-A;A-D;D-E;",
    # 'square-edge': "A-B;B-C;C-D;D-A;C-E;",
    # 'chordal-square-edge': "A-B;B-C;C-D;D-A;B-D;A-E;",
    # '4-clique-edge': "A-B;B-C;C-D;D-A;A-C;B-D;A-E;",
    # 6 nodes
    # '5-path': 'A-B;B-C;C-D;D-E;E-F;',
    # '5-star': 'A-B;A-C;A-D;A-E;A-F;',
    # 'quad-triangle': "A-B;B-C;C-D;D-E;E-F;F-A;B-D;B-E;B-F;",
    # 'triangle-core': "A-B;B-C;C-D;D-E;E-F;F-A;B-D;B-F;D-F;",
    # 'twin-c-square': "A-B;B-C;C-D;D-E;E-F;F-A;A-C;A-D;D-F;",
    # 'twin-clique-4': "A-B;B-C;C-D;D-E;E-F;F-A;A-C;B-F;C-E;C-F;D-F;",
    # 'star-of-david-plus': "A-B;B-C;C-D;D-E;E-F;F-A;A-C;A-E;B-D;B-F;C-E;C-F;D-F;",
    # 'b313': "A-B;B-C;C-A;C-D;D-E;D-F;E-F;",
    '6-clique': 'A-B;A-C;A-D;A-E;A-F;B-C;B-D;B-E;B-F;C-D;C-E;C-F;D-E;D-F;E-F;',
    '6-cycle': 'A-B;B-C;C-D;D-E;E-F;F-A;',
    # 7 nodes
    # '6-star': 'A-B;A-C;A-D;A-E;A-F;A-G;',
    # '7-clique': 'A-B;A-C;A-D;A-E;A-F;A-G;B-C;B-D;B-E;B-F;B-G;C-D;C-E;C-F;C-G;D-E;D-F;D-G;E-F;E-G;F-G;',
    # '7-cycle': 'A-B;B-C;C-D;D-E;E-F;F-G;G-A;',
    # 8 nodes
    # '8-clique': 'A-B;A-C;A-D;A-E;A-F;A-G;A-H;B-C;B-D;B-E;B-F;B-G;B-H;C-D;C-E;C-F;C-G;C-H;D-E;D-F;D-G;D-H;E-F;E-G;E-H;F-G;F-H;G-H;',
    # '8-cycle': 'A-B;B-C;C-D;D-E;E-F;F-G;G-H;H-A;',
    # 9 nodes
    # '9-clique': 'A-B;A-C;A-D;A-E;A-F;A-G;A-H;A-I;B-C;B-D;B-E;B-F;B-G;B-H;B-I;C-D;C-E;C-F;C-G;C-H;C-I;D-E;D-F;D-G;D-H;D-I;E-F;E-G;E-H;E-I;F-G;F-H;F-I;G-H;G-I;H-I;',
    # '9-cycle': 'A-B;B-C;C-D;D-E;E-F;F-G;G-H;H-I;I-A;',
    # 10 nodes
    # '10-clique': 'A-B;A-C;A-D;A-E;A-F;A-G;A-H;A-I;A-J;B-C;B-D;B-E;B-F;B-G;B-H;B-I;B-J;C-D;C-E;C-F;C-G;C-H;C-I;C-J;D-E;D-F;D-G;D-H;D-I;D-J;E-F;E-G;E-H;E-I;E-J;F-G;F-H;F-I;F-J;G-H;G-I;G-J;H-I;H-J;I-J;',
    # '10-cycle': 'A-B;B-C;C-D;D-E;E-F;F-G;G-H;H-I;I-J;J-A;',
})


def get_edge_list_filename(dataset_name, index):
    return f'{os.path.dirname(os.path.abspath(__file__))}/tmp/graphs/{dataset_name}_graph_{index}.txt'

def get_hom_count_filename(dataset_name, index, pattern_name):
    return f'{os.path.dirname(os.path.abspath(__file__))}/tmp/hom_counts/{dataset_name}_graph_{index}_{pattern_name}.csv'

def get_subgraph_count_filename(dataset_name, index, pattern_name):
    return f'{os.path.dirname(os.path.abspath(__file__))}/tmp/subgraph_counts/{dataset_name}_graph_{index}_{pattern_name}.csv'

def write_edge_list(dataset):
    for index, graph in enumerate(dataset):
        filename = get_edge_list_filename(dataset.name, index)
        if os.path.exists(filename):
            continue
        with open(filename, 'w') as f:
            for i, j in graph.edge_index.T:
                f.writelines(f'{i.item()} {j.item()}\n')

def write_hom_count(dataset):
    # java_home = subprocess.check_output(['/usr/libexec/java_home', '-v' '1.8'], text=True)
    new_env = os.environ.copy()
    # new_env['JAVA_HOME'] = java_home.strip()

    for pattern in patterns:
        query = patterns[pattern]
        for index, graph in enumerate(dataset):
            hom_file_name = get_hom_count_filename(dataset_name, index, pattern)
            if os.path.exists(hom_file_name):
                continue
            args = [
                './spark-2.4.8-bin-hadoop2.7/bin/spark-submit',
                '--class', 'org.apache.spark.disc.SubgraphCounting',
                '../DISC/DISC-assembly-0.1.jar',
                '-p', '../DISC/disc_local.properties',
                '-d', get_edge_list_filename(dataset_name, index),
                '-q', query, '-e', 'Result', '-u', 'HOM', '-c', 'A',
                '-o', hom_file_name
            ]
            subprocess.run(args, cwd='./', text=True, check=True, env=new_env)


if __name__ == '__main__':
    os.makedirs(f'./tmp/graphs', exist_ok=True)

    graph_dataset_names = [
        # 'MUTAG',
        # 'PTC_MR',
        # 'NCI1',
        # 'PROTEINS',
        # 'IMDB-BINARY',
        # 'IMDB-MULTI',
        # 'REDDIT-BINARY',
        # 'REDDIT-MULTI-5K',
        # 'COLLAB',
        'DD',
        # 'ENZYMES'
        'MCF-7'
    ]
    for dataset_name in graph_dataset_names:
        dataset = TUDataset('../data/TUDataset', dataset_name, use_node_attr=True)
        write_edge_list(dataset)

    # graph_dataset_names = ['synthetic_cycles']
    # for dataset_name in graph_dataset_names:
    #     dataset = SyntheticCycles('../data/', dataset_name)
    #     write_edge_list(dataset)

    # node_dataset_names = ['Cora', 'CiteSeer', 'PubMed']
    # for dataset_name in node_dataset_names:
    #     dataset = Planetoid('../data/Planetoid', dataset_name)
    #     write_edge_list(dataset)

    os.makedirs(f'./tmp/hom_counts', exist_ok=True)
    #
    for dataset_name in graph_dataset_names:
        dataset = TUDataset('../data/TUDataset', dataset_name, use_node_attr=True)
        write_hom_count(dataset)

    # for dataset_name in node_dataset_names:
    #     dataset = Planetoid('../data/Planetoid', dataset_name)
    #     write_hom_count(dataset)

    # graph_dataset_names = ['synthetic_cycles']
    # for dataset_name in graph_dataset_names:
    #     dataset = SyntheticCycles('../data/', dataset_name)
    #     write_hom_count(dataset)
