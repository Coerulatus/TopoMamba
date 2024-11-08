import os
import subprocess
target_gpu = 1
gpu_idx = min(target_gpu, len(subprocess.check_output(['nvidia-smi','-L']).decode('utf-8').strip().split('\n'))-1)
os.environ["CUDA_VISIBLE_DEVICES"]=f"{gpu_idx}"

import sys
import yaml
import time
import numpy as np
import wandb
import torch
import torch_geometric
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary, EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from topomamba.data.pl_data import PLDatamoduleBatch
from topomamba.io.load.loaders import GraphLoader
from topomamba.data.utils import DotDict
from topomamba.data.liftings import SimplicialCliqueLifting, SimplicialNeighborhoodLifting
from topomamba.models.topomamba import TopoMamba
from topomamba.evaluators.test import CustomTest


torch.set_float32_matmul_precision('high')
torch.set_num_threads(2)
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

def sweep_iteration():
    time_start = time.time()
    wandb_logger_test = wandb.init()
    pl.seed_everything(wandb.config.seed)
    
    if hasattr(wandb.config, 'year'):
        year = wandb.config.year
    else:
        year = None
    if hasattr(wandb.config, 'task_variable'):
        task_variable = wandb.config.task_variable
    else:
        task_variable = None
    data_params = DotDict({'data_domain': 'graph',
                           'data_type': wandb.config.data_type,
                           'data_name': wandb.config.data_name,
                           'data_dir': wandb.config.data_dir+'_complex_dim_'+str(wandb.config.complex_dim),
                           'data_split_dir': wandb.config.data_dir+'/splits',
                           'data_seed': wandb.config.split,
                           'year': year,
                           'task_variable': task_variable,
                           'split_type': 'random',
                           'k': 10,
                           'train_prop': 0.5,
                           'pin_memory': False})
    
    if hasattr(wandb.config, 'y_target'):
        y_target = wandb.config.y_target
    else:
        y_target = None
    if hasattr(wandb.config, 'task'):
        task = wandb.config.task
    else:
        task = "classification"
    if hasattr(wandb.config, 'backwards'):
        backwards = wandb.config.backwards
    else:
        backwards = True
    if hasattr(wandb.config, 'save_results'):
        save_results = wandb.config.save_results
    else:
        save_results = False
    if hasattr(wandb.config, 'hops'):
        hops = wandb.config.hops
    else:
        hops = 1
    if hasattr(wandb.config, 'max_k_simplices'):
        max_k_simplices = wandb.config.max_k_simplices
    else:
        max_k_simplices = 100000
    if hasattr(wandb.config, 'feature_lifting'):
        feature_lifting = wandb.config.feature_lifting
    else:
        feature_lifting = "sum"
    if hasattr(wandb.config, 'aggregation'):
        aggregation = wandb.config.aggregation
    else:
        aggregation = "sum"
    if hasattr(wandb.config, 'weight_decay'):
        weight_decay = wandb.config.weight_decay
    else:
        weight_decay = 0.0
        
    log = True
    # create dataset
    if hasattr(wandb.config, 'lifting') and wandb.config.lifting == "neighborhood":
        transform = SimplicialNeighborhoodLifting(complex_dim=wandb.config.complex_dim, max_k_simplices=max_k_simplices, hops=hops, signed=False)
    else:
        transform = SimplicialCliqueLifting(complex_dim=wandb.config.complex_dim, signed=False)


    graph_loader = GraphLoader(data_params, transform)
    dataset = graph_loader.load()
    datamodule = PLDatamoduleBatch(dataset[0], batch_size=wandb.config.batch_size, max_n_neighbors=100000, n_hops=wandb.config.n_layers, model='mamba')

    # create model
    d_input = datamodule.dataset.x.shape[1]
    if task == "classification":
        d_out = int(torch.max(datamodule.dataset.y).item() + 1)
    else:
        d_out = 1
        
    if task == "classification":
        loss = torch.nn.CrossEntropyLoss()
    elif task == "regression":
        loss = torch.nn.MSELoss()
        
    batch = next(iter(datamodule.train_dataloader()))
    if wandb.config.sequence_model == "mamba":
        if device == "cpu":
            raise ValueError("Mamba model is not supported on CPU")
        
    model = TopoMamba(d_input, 
                            d_out,
                            d_hidden=wandb.config.d_hidden,
                            n_layers=wandb.config.n_layers,
                            sequencer=wandb.config.sequencer,
                            sequencer_limit=wandb.config.sequencer_limit,
                            keep_empty_ranks=wandb.config.keep_empty_ranks,
                            same_graph=False,
                            task_level="node",
                            task=task,
                            loss=loss,
                            lr=wandb.config.lr,
                            weight_decay=weight_decay,
                            input_dropout=wandb.config.dropout,
                            readout_dropout=wandb.config.dropout,
                            skip_connection=wandb.config.skip_connection,
                            backwards=backwards,
                            device=device,
                            log=log,
                            feature_lifting=feature_lifting,
                            aggregation=aggregation,
                            sequence_model=wandb.config.sequence_model,
                            batch_size=wandb.config.batch_size,
                            save_results = save_results,
                            time_start=time_start)
   
    model.sequencer.setup(datamodule.dataset.incidence)
    if wandb.config.data_name == "questions":
        model.balance_loss(dataset[0].y)
        
    if task == "classification":
        if wandb.config.data_name == "minesweeper":
            metric_monitor = "valid_rocauc"
        else:
            metric_monitor = "valid_acc"
        mode = "max"
    else:
        metric_monitor = "valid_mae"
        mode = "min"
        
    early_stop_callback = EarlyStopping(monitor=metric_monitor, min_delta=0.00, patience=wandb.config.patience, verbose=False, mode=mode)
    checkpoint_callback = ModelCheckpoint(dirpath='checkpoints/', save_weights_only=True, mode="max", monitor=metric_monitor)
    wandb_logger = WandbLogger()
    trainer = pl.Trainer(accelerator="gpu",
                         devices=1,
                         max_epochs=wandb.config.n_epochs, 
                         callbacks=[checkpoint_callback,
                                    ModelSummary(max_depth=1),
                                    early_stop_callback],
                         log_every_n_steps=1,
                         check_val_every_n_epoch=1,
                         logger = wandb_logger)
    torch.cuda.reset_peak_memory_stats()
    trainer.fit(model, datamodule.train_dataloader(), datamodule.val_dataloader())
    
    model = TopoMamba.load_from_checkpoint(checkpoint_callback.best_model_path)
    model.sequencer.setup(datamodule.dataset.incidence)
    CustomTest(model, datamodule.test_dataloader(), wandb_logger_test)
    
if __name__ == "__main__":
    arguments = sys.argv
    if len(arguments) != 2:
        print("Usage: python train.py <sweep_config_file>")
        sys.exit(1)
    with open(arguments[1], "r") as file:
        sweep_config = yaml.safe_load(file)
    wandb.login() 
    sweep_id = wandb.sweep(sweep_config, project="debug")
    wandb.agent(sweep_id, function=sweep_iteration)
    wandb.finish()

