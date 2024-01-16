import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.graphgym.models.head  # noqa, register module
import torch_geometric.graphgym.register as register
import torch_geometric.nn as pyg_nn
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_network


@register_network('GCN')
class GNN(torch.nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        conv_model = pyg_nn.GCNConv
        if cfg.dataset.task == 'graph':
            self.pooling_fun = register.pooling_dict[cfg.model.graph_pooling]
            self.post_mp = nn.Sequential(
                nn.Linear(cfg.gnn.dim_inner, cfg.gnn.dim_inner),
                nn.BatchNorm1d(cfg.gnn.dim_inner, track_running_stats=False),
                nn.ReLU(inplace=True),
                nn.Linear(cfg.gnn.dim_inner, dim_out),
            )
        self.convs = nn.ModuleList()
        self.convs.append(conv_model(dim_in, cfg.gnn.dim_inner, bias=False, improved=False))
        if cfg.gnn.layers_mp == 2:
            if cfg.dataset.task == 'graph':
                self.convs.append(conv_model(cfg.gnn.dim_inner, cfg.gnn.dim_inner, bias=False, improved=False))
            else:
                self.convs.append(conv_model(cfg.gnn.dim_inner, dim_out, bias=False, improved=False))
        else:
            for _ in range(cfg.gnn.layers_mp - 2):
                self.convs.append(conv_model(cfg.gnn.dim_inner, cfg.gnn.dim_inner, bias=False, improved=False))
            if cfg.dataset.task == 'graph':
                self.convs.append(conv_model(cfg.gnn.dim_inner, cfg.gnn.dim_inner, bias=False, improved=False))
            else:
                self.convs.append(conv_model(cfg.gnn.dim_inner, dim_out, bias=False, improved=False))

    def _apply_index(self, batch):
        if cfg.dataset.task == 'graph':
            return batch.graph_feature, batch.y
        mask = '{}_mask'.format(batch.split)
        return batch.x[batch[mask]], \
            batch.y[batch[mask]]

    def forward(self, batch):
        x, edge_index = batch.x, batch.edge_index

        for i in range(len(self.convs)):
            x = self.convs[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=cfg.gnn.dropout, training=self.training)

        batch.x = x
        if cfg.dataset.task == 'graph':
            x = self.pooling_fun(batch.x, batch.batch)
            x = self.post_mp(x)
            batch.x = x
            return batch.x, batch.y
        else:
            pred, label = self._apply_index(batch)
            return pred, label
