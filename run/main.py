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
from torch_geometric.graphgym.train import GraphGymDataModule
from torch_geometric.graphgym.loader import create_dataset
from torch_geometric.graphgym.loader import get_loader
from train import train

set_cfg(cfg)
torch.set_float32_matmul_precision('medium')

if __name__ == '__main__':
    # Load cmd line args
    args = parse_args()
    # Load config file
    load_cfg(cfg, args)
    set_out_dir(cfg.out_dir, args.cfg_file)
    # Set Pytorch environment
    torch.set_num_threads(cfg.num_threads)
    dump_cfg(cfg)
    # Repeat for different random seeds
    for i in range(args.repeat):
        set_run_dir(cfg.out_dir)
        set_printing()
        # Set configurations for each run
        cfg.seed = cfg.seed + 1
        seed_everything(cfg.seed)
        auto_select_device()
        # Set machine learning pipeline
        datamodule = GraphGymDataModule()
        model = create_model(to_device=False)
        # Print model info
        logging.info(model)
        logging.info(cfg)
        cfg.params = params_count(model)
        logging.info('Num parameters: %s', cfg.params)
        train(model, datamodule, logger=True)

        logging.info('Run %d finished', i + 1)
        logging.info('Training diameters')
        dataset = create_dataset()
        id = dataset.data['train_graph_index']
        loader = get_loader(dataset[id], cfg.train.sampler,len(id))
        model.eval()
        pred, true = model(list(loader)[0])
        diameters = []
        dist = torch.cdist(pred, pred)
        # diameter across all classes
        diameters.append(dist.max().item())
        for y in range(true.max() + 1):
            mask = true == y
            if mask.sum() == 0:
                diameters.append(torch.nan)
                continue
            dist = torch.cdist(pred[mask], pred[mask])
            diameter = dist.max().item()
            diameters.append(diameter)
        logging.info('Diameter,' + ','.join(map(str,diameters)))

    # Aggregate results from different seeds
    agg_runs(cfg.out_dir, cfg.metric_best)
    # When being launched in batch mode, mark a yaml as done
    if args.mark_done:
        os.rename(args.cfg_file, f'{args.cfg_file}_done')
