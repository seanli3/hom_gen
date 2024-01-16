import pandas as pd
from torch_geometric.graphgym.utils.io import json_to_dict_list
import os
import numpy as np

dataset = 'CiteSeer'
type = 'GCN'
dim_in=64
c_type='HOM'
# patterns=['2-path,3-path,4-path','triangle,4-cycle,5-cycle','triangle,4-clique,5-clique']
patterns=['2-path,3-path,4-path,5-path','triangle,4-cycle,5-cycle,6-cycle','triangle,4-clique,5-clique,6-clique']

headers = ['epoch','params','time_iter']
values = []
# for dataset in ['TU_ENZYMES','TU_PROTEINS','TU_PTC_MR']:

print('dataset type l_mp dim_in c_type pattern', end=' ')
# print('dataset type l_mp dim_in', end=' ')
print(' '.join(headers), end=' ')
print('val_loss val_loss_std train_loss train_loss_std val_acc val_acc_std train_acc train_acc_std loss_gap loss_gap_std gap gap_std')

# for dataset in ['Cora','CiteSeer','PubMed']:
# for dataset in ['Cornell','Texas','Wisconsin']:
for dataset in ['TU_ENZYMES','TU_PROTEINS','Tu_PTC_MR']:
    for l_mp in range(1, 7):
        for pattern in patterns:
            dir = f'results/graph_grid_hom/graph-dataset={dataset}-type={type}-l_mp={l_mp}-dim_in={dim_in}-c_type={c_type}-add_counts=True-patterns={pattern}'
        # dir = f'results/node_grid_node/node-dataset={dataset}-type={type}-l_mp={l_mp}-dim_in={dim_in}'
            gaps =[]
            loss_gaps =[]
            val_accuracy = []
            train_accuracy = []
            val_loss = []
            train_loss = []
            # for run in [41,42,45]:
            # for run in range(41,46):
            for run in range(0,5):
                train_stats = pd.DataFrame(json_to_dict_list(os.path.join(dir, '{}'.format(run), 'train', 'stats.json')))
                best_epoch = train_stats['accuracy'].idxmax()
                # best_epoch = train_stats['loss'].idxmin()
                best_train = train_stats.iloc[best_epoch]
                best_train_acc = best_train['accuracy']
                best_train_loss = best_train['loss']

                val_stats = pd.DataFrame(json_to_dict_list(os.path.join(dir, '{}'.format(run), 'val', 'stats.json')))
                # best_epoch = val_stats['accuracy'].idxmax()
                # best_epoch = val_stats['loss'].idxmin()
                best_val = val_stats.iloc[best_epoch]
                best_val_acc = best_val['accuracy']
                best_val_loss = best_val['loss']

                loss_gaps.append(best_train_loss - best_val_loss)
                gaps.append(best_train_acc - best_val_acc)
                val_accuracy.append(best_val_acc)
                train_accuracy.append(best_train_acc)
                val_loss.append(best_val_loss)
                train_loss.append(best_train_loss)

            print(f'{dataset} {type} {l_mp} {dim_in} {c_type} {pattern}', end=' ')
            # print(f'{dataset} {type} {l_mp} {dim_in}', end=' ')
            for header in headers:
                print(f'{best_val[header]}', end=' ')
            print(f'{np.mean(val_loss)} {np.std(val_loss)} {np.mean(train_loss)} {np.std(train_loss)} {np.mean(val_accuracy)} {np.std(val_accuracy)} {np.mean(train_accuracy)} {np.std(train_accuracy)} {np.mean(loss_gaps)} {np.std(loss_gaps)} {np.mean(gaps)} {np.std(gaps)}')
