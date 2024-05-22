import time
from sklearn.metrics import mean_absolute_error
from typing import Any, Dict, Tuple

import torch

from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.imports import LightningModule
from torch_geometric.graphgym.loss import compute_loss
from torch_geometric.graphgym.models.gnn import GNN
from torch_geometric.graphgym.optim import create_optimizer, create_scheduler
from torch_geometric.graphgym.register import network_dict, register_network

register_network('gnn_cus', GNN)


class GraphGymModule(LightningModule):
    def __init__(self, dim_in, dim_out, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = network_dict[cfg.model.type](dim_in=dim_in,
                                                  dim_out=dim_out)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def configure_optimizers(self) -> Tuple[Any, Any]:
        optimizer = create_optimizer(self.model.parameters(), self.cfg.optim)
        scheduler = create_scheduler(optimizer, self.cfg.optim)
        return [optimizer], [scheduler]

    def _shared_step(self, batch, split: str) -> Dict:
        batch.split = split
        pred, true = self(batch)
        loss, pred_score = compute_loss(pred, true)
        step_end_time = time.time()
        mae = None
        if 'regression' in cfg.dataset.task_type:
            mae = float(round(mean_absolute_error(true.detach().cpu(), pred.detach().cpu()), cfg.round))
        return dict(loss=loss, true=true, pred_score=pred_score.detach(),
                step_end_time=step_end_time, mae=mae)

    def training_step(self, batch, *args, **kwargs):
        res = self._shared_step(batch, split="train")
        self.log('train_loss', res['loss'], batch_size=len(batch))
        if 'regression' in cfg.dataset.task_type:
            self.log('train_mae', res['mae'], batch_size=len(batch))
        return res

    def validation_step(self, batch, *args, **kwargs):
        res = self._shared_step(batch, split="val")
        self.log('val_loss', res['loss'], batch_size=len(batch))
        if 'regression' in cfg.dataset.task_type:
            self.log('val_mae', res['mae'], batch_size=len(batch))
        return res

    def test_step(self, batch, *args, **kwargs):
        return self._shared_step(batch, split="test")

    @property
    def encoder(self) -> torch.nn.Module:
        return self.model.encoder

    @property
    def mp(self) -> torch.nn.Module:
        return self.model.mp

    @property
    def post_mp(self) -> torch.nn.Module:
        return self.model.post_mp

    @property
    def pre_mp(self) -> torch.nn.Module:
        return self.model.pre_mp


def create_model(to_device=True, dim_in=None, dim_out=None) -> GraphGymModule:
    r"""Create model for graph machine learning.

    Args:
        to_device (string): The devide that the model will be transferred to
        dim_in (int, optional): Input dimension to the model
        dim_out (int, optional): Output dimension to the model
    """
    dim_in = cfg.share.dim_in if dim_in is None else dim_in
    dim_out = cfg.share.dim_out if dim_out is None else dim_out

    model = GraphGymModule(dim_in, dim_out, cfg)
    if to_device:
        model.to(torch.device(cfg.accelerator))
    return model
