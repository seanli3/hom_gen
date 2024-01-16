import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_excel('./hom_gen.xlsx', sheet_name='Node New Split')
data['patterns'] = '1-WL'
no_hom_data = data[data['type']=='GCN']
no_hom_data = no_hom_data.iloc[no_hom_data.groupby(['dataset','patterns', 'l_mp'])['val_accuracy'].idxmax()]

hom_data = pd.read_excel('./hom_gen.xlsx', sheet_name='Node Hom New Split New')
hom_data = hom_data[hom_data['type']=='GCN']
hom_data = hom_data.iloc[hom_data.groupby(['dataset','patterns','l_mp'])['val_accuracy'].idxmax()]
best_val_record = pd.concat([no_hom_data, hom_data])

bounds = pd.read_csv('./new_bounds.csv', delimiter=' ')
agg_bounds = bounds.groupby(['patterns','dataset','iteration']).agg(bound_mean = ('bound', 'mean'), bound_std = ('bound', 'std')).reset_index()
agg_bounds['type'] = 'Bounds'
agg_bounds.rename({'iteration':'l_mp'}, axis=1, inplace=True)

merged_df = best_val_record.merge(agg_bounds, on=['dataset','patterns', 'l_mp'], how='left')

def plot(dataset, y1lim, y2lim, patterns):
    disp_df = merged_df[(merged_df['dataset'] == dataset) & (merged_df['patterns'] == patterns)]
    disp_df = disp_df.set_index('l_mp')
    fig, ax1 = plt.subplots()
    ax = disp_df.gap.plot(kind='bar', color='green', ax=ax1, width=.3, position=1)
    x_coords = [p.get_x() + 0.5 * p.get_width() for p in ax.patches]
    y_coords = [p.get_height() for p in ax.patches]
    ax.errorbar(x=x_coords, y=y_coords, yerr=disp_df["gap_std"], fmt="none", c="k")
    # ax = sns.barplot(disp_df, x="patterns", y="gap", hue="type",ax=ax1)
    ax2 = ax1.twinx()
    ax = disp_df.bound_mean.plot(kind='bar', color='blue', x='l_mp', ax=ax2, width=0.3, position=0)
    x_coords = [p.get_x() + 0.5 * p.get_width() for p in ax.patches]
    y_coords = [p.get_height() for p in ax.patches]
    ax.errorbar(x=x_coords, y=y_coords, yerr=disp_df["bound_std"], fmt="none", c="k")
    ax1.set_ylabel('gap')
    ax1.set_ylim(y1lim)
    ax2.set_ylim(y2lim)
    ax2.set_ylabel('bound')
    ax.set_title(dataset)

    plt.show()

plot('Cora', (0.01, 0.15), (0.0, 0.04), '1-WL')
plot('CiteSeer', (0.0, 0.2), (0.0, 0.022), '1-WL')
plot('PubMed', (0.0, 0.2), (0.0, 0.013), '1-WL')

## sns.lineplot(data=merged_df[merged_df['patterns']=='1-WL'], x='l_mp', y='gap', hue='dataset')
#dip_df = merged_df[merged_df['patterns']=='1-WL']
#fig, ax1 = plt.subplots()
#dip_df[dip_df['dataset']=='Cora'].plot('l_mp', 'gap', yerr='gap_std', subplots=False, title='1-WL', ax=ax1, kind='line', color='red')
## dip_df[dip_df['dataset']=='CiteSeer'].plot('l_mp', 'gap', yerr='gap_std', subplots=False, title='1-WL', ax=ax1, kind='line', color='blue')
## dip_df[dip_df['dataset']=='PubMed'].plot('l_mp', 'gap', yerr='gap_std', subplots=False, title='1-WL', ax=ax1, kind='line', color='orange')
#ax2 = ax1.twinx()
#dip_df[dip_df['dataset']=='Cora'].plot('l_mp', 'bound_mean', yerr='bound_std', subplots=False, title='1-WL', ax=ax2, kind='line', color='pink')
## dip_df[dip_df['dataset']=='CiteSeer'].plot('l_mp', 'bound_mean', yerr='bound_std', subplots=False, title='1-WL', ax=ax2, kind='line', color='cyan')
## dip_df[dip_df['dataset']=='PubMed'].plot('l_mp', 'bound_mean', yerr='bound_std', subplots=False, title='1-WL', ax=ax2, kind='line', color='yellow')
#ax1.set_ylim(0,0.4)
#ax2.set_ylim(0,0.038)
#plt.show()