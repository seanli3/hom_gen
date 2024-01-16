from homcount.src.ghc.utils.HomSubio import HomSub, write_PACE_graphs
import tempfile
import os
from matplotlib import pyplot as plt

graph_directory = tempfile.mkdtemp()

import networkx as nx
from collections import OrderedDict
import sys
import subprocess

patterns = [
    {
        'name': '1-path',
        'graph': nx.path_graph(1),
        'td': 's td 1 1 1\nb 1 1\n'
    },{
        'name': '2-path',
        'graph': nx.path_graph(2),
        'td': 's td 1 2 2\nb 1 1 2\n'
    }, {
        'name': '3-path',
        'graph': nx.path_graph(3),
        'td': 's td 2 2 3\nb 1 1 2\nb 2 2 3\n1 2\n'
    }, {
       'name': '4-path',
        'graph': nx.path_graph(4),
        'td': 's td 3 2 4\nb 1 1 2\nb 2 2 3\nb 3 3 4\n1 2\n2 3\n',
    }, {
        'name': '5-path',
        'graph': nx.path_graph(5),
        'td': 's td 4 2 5\nb 1 1 2\nb 2 3 4\nb 3 2 3\nb 4 4 5\n1 3\n2 3\n2 4\n',
    }, {
        'name': '4-star',
        'graph': nx.star_graph(4),
        'td': 's td 3 2 4\nb 1 1 2\nb 2 2 3\nb 3 2 4\n1 2\n1 3\n',
    }, {
        'name': '3-cycle',
        'graph': nx.cycle_graph(3),
        'td': 's td 1 3 3\nb 1 1 2 3\n',
    }, {
        'name': '4-cycle',
        'graph': nx.cycle_graph(4),
        'td': 's td 2 3 4\nb 1 1 2 3\nb 2 1 3 4\n1 2\n'
    }, {
        'name': 'house',
        'graph': nx.house_graph(),
        'td': 's td 3 3 5\nb 1 1 2 3\nb 2 3 4 5\nb 3 1 3 4\n1 3\n2 3\n'
    }, {
        'name': '4-clique',
        'graph': nx.complete_graph(4),
        'td': 's td 1 4 4\nb 1 1 2 3 4\n'
    }
    # , {
    #     'name': '5-clique',
    #     'graph': nx.complete_graph(5),
    #     'td': 's td 1 5 5\nb 1 1 2 3 4 5\n'
    # }
]

pattern_list=list(map(lambda p:p['graph'], patterns))
from torch_geometric.utils import to_networkx, to_undirected
from fair_comparison.datasets import Enzymes, Proteins, IMDBBinary

dataset = IMDBBinary()
graph_list = [to_networkx(dataset.dataset[i]).to_undirected() for i in range(len(dataset.dataset))]

write_PACE_graphs(pattern_list, folder=graph_directory, prefix='pattern')

cwd = './'

ngraphs = len(graph_list)
npatterns = len(pattern_list)
td_list=list(map(lambda p:p['td'], patterns))

min_gsize = 999999
min_gidx = 0
for i in range(ngraphs):
    if graph_list[i].number_of_nodes() < min_gsize:
        min_gsize = graph_list[i].number_of_nodes()
        min_gidx = i

ig = 0
jp = 7

# g = graph_list[ig]
g = nx.path_graph(4).to_undirected()

nx.draw_networkx(g)

plt.show()
write_PACE_graphs([g], folder=graph_directory, prefix='graph')

print('graph size', str(g.number_of_nodes()))
print('pattern size', str(pattern_list[jp].number_of_nodes()))
with open('tam.out', 'w') as td_file:
    td_file.write(td_list[jp])
args = ['/Users/sohey/git/Substructure/homcount/HomSub/experiments-build/experiments/experiments',
        '-count-hom',
        '-h', os.path.join(graph_directory, f'pattern_{jp}.gr'),
        '-g', os.path.join(graph_directory, f'graph_{ig}.gr')]
report = subprocess.run(args, cwd=cwd, stdout=sys.stdout, stderr=sys.stderr, text=True, check=True)
