import torch

from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_loss


@register_loss('smoothl1')
def loss_example(pred, true):
    if cfg.model.loss_fun == 'smoothl1':
        l1_loss = torch.nn.SmoothL1Loss()
        loss = l1_loss(pred, true)
        return loss, pred

@register_loss('margin')
def loss_example(pred, true):
    if cfg.model.loss_fun == 'margin':
        pred = torch.nn.functional.log_softmax(pred, dim=-1)
        multi_margin_loss = torch.nn.MultiMarginLoss(p=1, margin=1.0)
        loss = multi_margin_loss(pred, true)
        return loss, pred



