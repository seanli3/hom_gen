import math
from itertools import chain

import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

def read_data(gap_sheet='Node New Split New Gap', hom_gap_sheet='Node Hom New Split New',
              bound_file='./new_bounds.csv', pattern='1-WL', type='GCN'):
    data = pd.read_excel('./hom_gen.xlsx', sheet_name=gap_sheet)
    data['pattern'] = pattern
    no_hom_data = data[data['type']=='GCN']
    # no_hom_data = no_hom_data.iloc[no_hom_data.groupby(['dataset','pattern', 'l_mp'])['val_acc'].idxmax()]

    hom_data = pd.read_excel('./hom_gen.xlsx', sheet_name=hom_gap_sheet)
    hom_data = hom_data[hom_data['type']==type]
    hom_data = hom_data.iloc[hom_data.groupby(['dataset','pattern','l_mp'])['val_acc'].idxmax()]
    best_val_record = pd.concat([no_hom_data, hom_data])

    bounds = pd.read_csv(bound_file, delimiter=' ')
    # bounds2 = pd.read_csv('./new_bounds.csv', delimiter=' ')
    # bounds = pd.concat([bounds1, bounds2])
    # agg_bounds = bounds.groupby(['pattern','dataset','iteration']).agg(bound_mean = ('bound', 'mean'), bound_std = ('bound', 'std')).reset_index()
    # agg_bounds['type'] = 'Bounds'
    bounds.rename({'iteration':'l_mp', 'patterns': 'pattern'}, axis=1, inplace=True)

    merged_df = best_val_record.merge(bounds, on=['dataset','pattern', 'l_mp'], how='left')
    merged_df.gap = merged_df.gap*100
    merged_df.gap_std = merged_df.gap_std*100
    merged_df.train_acc = merged_df.train_acc*100
    merged_df.val_acc = merged_df.val_acc*100
    return merged_df

def plot(df, dataset, y1lim, y2lim, pattern):
    disp_df = df[(df['dataset'] == dataset) & (df['pattern'] == pattern)]
    disp_df = disp_df.set_index('l_mp')
    fig, ax1 = plt.subplots()
    ax = disp_df.gap.plot(kind='bar', color='green', ax=ax1, width=.3, position=1)
    x_coords = [p.get_x() + 0.5 * p.get_width() for p in ax.patches]
    y_coords = [p.get_height() for p in ax.patches]
    ax.errorbar(x=x_coords, y=y_coords, yerr=disp_df["gap_std"], fmt="none", c="k")
    # ax = sns.barplot(disp_df, x="pattern", y="gap", hue="type",ax=ax1)
    ax2 = ax1.twinx()
    ax = disp_df.bound.plot(kind='bar', color='blue', x='l_mp', ax=ax2, width=0.3, position=0)
    x_coords = [p.get_x() + 0.5 * p.get_width() for p in ax.patches]
    y_coords = [p.get_height() for p in ax.patches]
    ax.errorbar(x=x_coords, y=y_coords, yerr=disp_df["bound_std"], fmt="none", c="k")
    ax1.set_ylabel('gap')
    ax1.set_ylim(y1lim)
    ax2.set_ylim(y2lim)
    ax2.set_ylabel('bound')
    ax.set_title(f'{dataset}-{pattern}')

    plt.show()

def plot_layers(df, save=False, datasets=['PubMed', 'Cora', 'CiteSeer'], y1lim=(0.01, 0.25), y2lim=(0.0, 1.6), layers=np.arange(1,7)):
    if save:
        plt.switch_backend('agg')
    disp_df = df[(df.pattern == '1-WL') & (df.l_mp.isin(layers))]
    fig = plt.figure(figsize=(4, 3))
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()
    x = np.arange(disp_df.l_mp.min(),disp_df.l_mp.max()+1)
    width = 0.15
    gap = 0.16
    colors = plt.get_cmap('Set2')
    lns = [[],[]]
    labels = [[],[]]
    for d in range(len(datasets)):
        lns1 = ax1.bar(
            x - math.floor(len(datasets)/2)*gap + gap*d,
            disp_df[disp_df.dataset==datasets[d]].gap,
            width,
            yerr=disp_df[disp_df.dataset==datasets[d]].gap_std,
            hatch=None, color=colors(d))
        lns2 = ax2.bar(
            x + math.ceil(len(datasets)/2)*gap + gap*d,
            disp_df[disp_df.dataset==datasets[d]].bound,
            width,
            yerr=disp_df[disp_df.dataset==datasets[d]].bound_std,
            hatch='\\', color=colors(d), alpha=0.8)
        lns[0].append(lns1[0])
        lns[1].append(lns2[0])
        labels[0].append(f'{datasets[d]} gap')
        labels[1].append(f'{datasets[d]} bound')

    ax1.legend(list(chain(*lns)), list(chain(*labels)), loc="upper left", fontsize=10)
    ax1.set_ylabel('Empirical gap', fontsize=11)
    ax1.set_ylim(y1lim)
    ax1.set_xticks(layers)
    ax1.set_xlabel('Layer', fontsize=11)
    ax1.set_yticks(np.arange(y1lim[0], y1lim[1], math.ceil((y1lim[1]-y1lim[0])/4*10)/10))
    ax2.set_ylim(y2lim)
    ax2.set_ylabel('Bound', fontsize=11)

    ax1.tick_params(axis='x', labelsize=10)
    ax1.tick_params(axis='y', labelsize=10)
    ax2.tick_params(axis='y', labelsize=10)

    fig.tight_layout()
    if save:
        plt.savefig('./img/graph_1wl_bounds.pdf', bbox_inches='tight', format='pdf')
    else:
        plt.show()



def plot_patterns(df, save=False, dataset='ENZYMES', patterns=[
    '1-WL',
        '2-path,3-path,4-path,5-path',
        'triangle,4-clique,5-clique,6-clique',
        'triangle,4-cycle,5-cycle,6-cycle',
    ],
      y1lim=(0.01, 0.25), y2lim=(0.0, 1.6), layer=4, legends=[
            r'$P_2$,$P_3$,$P_4$,$P_5$',
            r'$K_3$,$K_4$,$K_5$,$K_6$',
            r'$C_3$,$C_4$,$C_5$,$C_6$',
            r'$P_2$,$P_3$,$P_4$,$P_5$',
            r'$K_3$,$K_4$,$K_5$,$K_6$',
            r'$C_3$,$C_4$,$C_5$,$C_6$',
        ], legend=True):
    if save:
        plt.switch_backend('agg')
    disp_df = df[df.pattern.isin(patterns) & (df.dataset == dataset) & (df.l_mp == layer)].reset_index()
    fig = plt.figure(figsize=(4, 3))
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()
    x = 0
    width = 0.15
    gap = 0.16
    colors = plt.get_cmap('Set2')
    lns = [[],[]]
    labels = [[],[]]

    for d in range(len(patterns)):
        lns1 = ax1.bar(
            x - gap + gap*d,
            disp_df[disp_df.pattern==patterns[d]].gap,
            width,
            yerr=disp_df[disp_df.pattern==patterns[d]].gap_std,
            hatch=None, color=colors(d))
        lns2 = ax2.bar(
            x + 2*math.ceil(len(patterns)/2)*gap + gap*d + 0.1,
            disp_df[disp_df.pattern==patterns[d]].bound,
            width,
            yerr=disp_df[disp_df.pattern==patterns[d]].bound_std,
            hatch='\\', color=colors(d), alpha=0.8)
        lns[0].append(lns1[0])
        lns[1].append(lns2[0])
        labels[0].append(f'{patterns[d]} gap')
        labels[1].append(f'{patterns[d]} bound')

    if legend:
        ax1.legend(list(chain(*lns))[2:4], legends, loc="upper right", fontsize=15, ncols=1)
    # ax1.set_ylabel('Acc. gap', fontsize=15)
    ax1.set_ylim(y1lim)
    ax1.tick_params(axis='x', labelsize=14)
    ax1.tick_params(axis='y', labelsize=14)
    ax2.tick_params(axis='y', labelsize=14)
    ax1.set_yticks(np.arange(y1lim[0], y1lim[1], math.ceil((y1lim[1]-y1lim[0])/4*10)/10))
    ax2.set_ylim(y2lim)
    ax2.set_yticks(np.arange(y2lim[0], y2lim[1], math.ceil((y2lim[1]-y2lim[0])/4*10)/10))
    ax2.set_ylabel('Bound value', fontsize=15)

    ax1.set_xticks([-0.32,-0.16,0.,0.16])
    def format_fn(tick_val, tick_pos):
        if tick_pos == 1:
            return 'Gap'
        if tick_pos == 2:
            return 'Bound'
        else:
            return None

    from matplotlib.ticker import MaxNLocator
    ax1.xaxis.set_major_formatter(format_fn)
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    fig.tight_layout()
    if save:
        plt.savefig('./img/graph_patterns.pdf', bbox_inches='tight', format='pdf')
    else:
        plt.show()


# df = read_data(gap_sheet='Node New Split New Gap', hom_gap_sheet='Node Hom New Split New',
#                bound_file='./node_bounds.csv', pattern='1-WL', type='GCN')
# plot_layers(df, datasets=['PubMed', 'Cora', 'CiteSeer'], y1lim=(0., 0.5), y2lim=(0.0, 0.8))
# plot_patterns(df, save=True, y1lim=(10, 16), y2lim=(0, 4), layer=4, dataset='PubMed',  legends=[
#              'None',
#              r'$P_2$,$P_3$,$P_4$,$P_5$',
#              r'$K_3$,$K_4$,$K_5$,$K_6$',
#              r'$C_3$,$C_4$,$C_5$,$C_6$',
#          ])


# df = read_data(gap_sheet='Graph 0.9-0.1 split', hom_gap_sheet='Graph 0.9-0.1 Hom',
#                bound_file='./graph_bounds.csv', pattern='1-WL', type='GCN')
# plot_layers(df, save=False, datasets=['PROTEINS', 'ENZYMES'], y1lim=(0.0, 1), y2lim=(0.0, 27), layers=np.arange(1,6))
# plot_patterns(df, save=False, y1lim=(0., 68), y2lim=(0, 84), layer=4, dataset='ENZYMES',
# legends=[
#             'None',
#             r'$P_2$,$P_3$,$P_4$,$P_5$',
#     ])
# plot_patterns(df, save=True, y1lim=(0., 68), y2lim=(0, 84), layer=4, dataset='PROTEINS', legends=[
#             r'$K_3$,$K_4$,$K_5$,$K_6$',
#             r'$C_3$,$C_4$,$C_5$,$C_6$',
#         ])


# plot('Cora', (0.01, 0.25), (0.0, 1.6), '1-WL')
# plot('CiteSeer', (0.0, 0.4), (0.0, 1.4), '1-WL')
# plot('PubMed', (0.0, 0.2), (0.0, 0.3), '1-WL')

# plot('Cora', (0.05, 0.15), (0.0, 1.8), '2-path,3-path,4-path,5-path')
# plot('Cora', (0.05, 0.15), (0.0, 1.8), 'triangle,4-cycle,5-cycle,6-cycle')
# plot('Cora', (0.05, 0.15), (0.0, 1.8), 'triangle,4-clique,5-clique,6-clique')

# plot('CiteSeer', (0.1, 0.4), (0.1, 0.8), '2-path,3-path,4-path,5-path')
# plot('CiteSeer', (0.1, 0.4), (0.1, 0.8), 'triangle,4-cycle,5-cycle,6-cycle')
# plot('CiteSeer', (0.1, 0.4), (0.1, 0.8), 'triangle,4-clique,5-clique,6-clique')

# plot('PubMed', (0.0, 0.2), (0.0, 0.2), '2-path,3-path,4-path,5-path')
# plot('PubMed', (0.0, 0.2), (0.0, 0.2), 'triangle,4-cycle,5-cycle,6-cycle')
# plot('PubMed', (0., 0.2), (0.0, 0.2), 'triangle,4-clique,5-clique,6-clique')

## sns.lineplot(data=merged_df[merged_df['pattern']=='1-WL'], x='l_mp', y='gap', hue='dataset')
#dip_df = merged_df[merged_df['pattern']=='1-WL']
#fig, ax1 = plt.subplots()
#dip_df[dip_df['dataset']=='Cora'].plot('l_mp', 'gap', yerr='gap_std', subplots=False, title='1-WL', ax=ax1, kind='line', color='red')
## dip_df[dip_df['dataset']=='CiteSeer'].plot('l_mp', 'gap', yerr='gap_std', subplots=False, title='1-WL', ax=ax1, kind='line', color='blue')
## dip_df[dip_df['dataset']=='PubMed'].plot('l_mp', 'gap', yerr='gap_std', subplots=False, title='1-WL', ax=ax1, kind='line', color='orange')
#ax2 = ax1.twinx()
#dip_df[dip_df['dataset']=='Cora'].plot('l_mp', 'bound', yerr='bound_std', subplots=False, title='1-WL', ax=ax2, kind='line', color='pink')
## dip_df[dip_df['dataset']=='CiteSeer'].plot('l_mp', 'bound', yerr='bound_std', subplots=False, title='1-WL', ax=ax2, kind='line', color='cyan')
## dip_df[dip_df['dataset']=='PubMed'].plot('l_mp', 'bound', yerr='bound_std', subplots=False, title='1-WL', ax=ax2, kind='line', color='yellow')
#ax1.set_ylim(0,0.4)
#ax2.set_ylim(0,0.038)
#plt.show()