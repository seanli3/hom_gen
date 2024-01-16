from fair_comparison.datasets import Enzymes
from util import find_all_cycles
from torch_geometric.utils import to_networkx, to_undirected

enzymes = Enzymes()

f = open('enzymes.csv', 'w')
f.writelines('#graph,' + ','.join(['cycle length={}'.format(n) for n in range(3,11)]) + ',# nodes,# edges,max degree,y\n')
for i in range(len(enzymes.dataset.data)):
    G = to_networkx(enzymes.dataset.data[i])
    G = G.to_undirected()
    cycles = find_all_cycles(G, max_cycle_length=10, min_cycle_length=3)
    f.writelines([
        ('{},' + ','.join(['{}' for n in range(3,11)]) + ',{},{},{},{}\n').format(
            i,
            *[len(list(filter(lambda c:len(c) == n, cycles))) for n in range(3,11)],
            G.number_of_nodes(),
            G.number_of_edges(),
            max(dict(G.degree()).values()),
            enzymes.dataset.data[i].y.item()
        )
    ])

f.close()