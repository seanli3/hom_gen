import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.graphgym.models.head  # noqa, register module
import torch_geometric.graphgym.register as register
import torch_geometric.nn as pyg_nn
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_network


@register_network('GIN')
class GIN(torch.nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        conv_model = pyg_nn.GINConv
        self.convs = nn.ModuleList()

        mlp = nn.Linear(dim_in, dim_in)
        self.convs.append(conv_model(nn=mlp, dim_in=dim_in, dim_out=dim_in, train_eps=True))

        for _ in range(cfg.gnn.layers_mp - 1):
            mlp = nn.Linear(dim_in, dim_in)
            self.convs.append(conv_model(nn=mlp, dim_in=dim_in, dim_out=dim_in, train_eps=True))

        GNNHead = register.head_dict[cfg.dataset.task]
        self.post_mp = GNNHead(dim_in=dim_in, dim_out=dim_out)

    def forward(self, batch):
        x, edge_index = batch.x, batch.edge_index

        for i in range(len(self.convs)):
            x = self.convs[i](x, edge_index)
            x = F.dropout(x, p=cfg.gnn.dropout, training=self.training)

        batch.x = x
        batch = self.post_mp(batch)

        return batch
