import logging
import os
import torch
import custom_graphgym  # noqa, register custom modules

from torch_geometric import seed_everything
from torch_geometric.graphgym.cmd_args import parse_args
from torch_geometric.graphgym.config import (
    cfg,
    dump_cfg,
    load_cfg,
    set_out_dir,
    set_run_dir,
)
from torch_geometric.graphgym.logger import set_printing
from model_builder import create_model
from torch_geometric.graphgym.utils.agg_runs import agg_runs
from torch_geometric.graphgym.utils.comp_budget import params_count
from torch_geometric.graphgym.utils.device import auto_select_device
from torch_geometric.graphgym.config import set_cfg
from torch_geometric.graphgym.train import train, GraphGymDataModule
from wl.compute_bound import lipschitz

set_cfg(cfg)
torch.set_float32_matmul_precision('medium')

if __name__ == '__main__':
    # Load cmd line args
    args = parse_args()
    # Load config file
    load_cfg(cfg, args)
    # set_out_dir(cfg.out_dir, args.cfg_file)
    # Set Pytorch environment
    torch.set_num_threads(cfg.num_threads)
    dump_cfg(cfg)
    # set_run_dir(cfg.out_dir)
    set_printing()
    seed_everything(cfg.seed)
    auto_select_device()
    datamodule = GraphGymDataModule()
    model = create_model(to_device=False)
    # Print model info
    logging.info(model)
    logging.info(cfg)
    cfg.params = params_count(model)
    logging.info('Num parameters: %s', cfg.params)
    train(model, datamodule, logger=True)
    batch = list(datamodule.loaders[0])[0]
    batch.split = 'train'
    print(lipschitz(model.model, batch))

