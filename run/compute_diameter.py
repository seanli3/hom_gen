import glob
import custom_graphgym  # noqa, register custom modules
from torch_geometric.graphgym.cmd_args import parse_args
import numpy as np

if __name__ == '__main__':
    patterns = ['','2-path,3-path,4-path,5-path', 'triangle,4-cycle,5-cycle,6-cycle',
                'triangle,4-clique,5-clique,6-clique']
    # patterns = ['','triangle','4-cycle','chordal-square','4-clique','5-cycle','5-clique','6-clique','6-cycle']
    for dataset_name in ['TU_PROTEINS','TU_ENZYMES']:
        for p in patterns:
            for l in range(6,7):
                diameters = []
                for i in range(5):
                    if p == '':
                        file_path = glob.glob(f'results/graph_grid_layers/graph-l_mp={l}-dataset={dataset_name}/{i}/logging.log')[0]
                    else:
                        file_path = glob.glob(f'results/graph_grid_hom/graph-dataset={dataset_name}-model=GCN-c_type=HOM-l_mp={l}-add_counts=True-patterns={p}/{i}/logging.log')[0]
                    with open(file_path, 'r') as f:
                        lines = f.readlines()
                        if 'Diameter' in lines[-1]:
                            last_line = lines[-1]
                        else:
                            last_line = lines[-2]
                    diameters.append(list(map(float, last_line.split(',')[1:])))
                diameters = np.array(diameters)
                print('{},{},diameter mean,{} layer,{},{}'.format(
                    'GCN'+ ('' if p == '' else f'-{p}'),
                    dataset_name, l, ','.join(map(str, diameters.mean(axis=0).tolist())),
                    (diameters / diameters[:, 0].reshape(-1, 1))[:, 1:].mean(1).mean())
                )
                print('{},{},diameter std,{} layer,{},{}'.format(
                    'GCN'+ ('' if p == '' else f'-{p}'),
                    dataset_name, l, ','.join(map(str, diameters.std(axis=0).tolist())),
                      (diameters / diameters[:, 0].reshape(-1, 1))[:, 1:].mean(1).std())
                )
                # print('{},{},diameter mean,{} layer,{}'.format('GCN', dataset_name, l,
                #                                                ','.join(map(str, diameters.mean(axis=0).tolist()))))
                # print('{},{},diameter std,{} layer,{}'.format('GCN', dataset_name, l,
                #                                           ','.join(map(str, diameters.std(axis=0).tolist()))))
