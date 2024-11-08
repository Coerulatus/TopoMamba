import sys
import yaml
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
from topomamba.data.liftings import SimplicialCliqueLifting
from topomamba.models.topomamba import TopoMamba
from topomamba.evaluators.test import CustomTest

data_types = ['cocitation','cocitation','cocitation','cornel','heterophilic','heterophilic','heterophilic']
data_names = ['cora','citeseer','pubmed','US-county-demos','minesweeper','amazon_ratings','roman_empire']
data_dirs = ['data_temp/cora','data_temp/citeseer','data_temp/pubmed','data_temp/uscountydemos','data_temp/minesweeper','data_temp/amazonratings','data_temp/romanempire']
split = 0
years = [None,None,None,2012,None,None,None]
task_variables = [None,None,None,'Election',None,None,None]

for data_type, data_name, data_dir, year, task_variable in zip(data_types, data_names, data_dirs, years, task_variables):
    data_params = DotDict({'data_domain': 'graph',
                            'data_type': data_type,
                            'data_name': data_name,
                            'data_dir': data_dir,
                            'data_split_dir': data_dir+'/splits',
                            'data_seed': split,
                            'year': year,
                            'task_variable': task_variable,
                            'split_type': 'random',
                            'k': 10,
                            'train_prop': 0.5,
                            'pin_memory': False})
    transform = SimplicialCliqueLifting(complex_dim=4, signed=False)
    graph_loader = GraphLoader(data_params, transform)
    dataset = graph_loader.load()
    data = dataset[0]
    print(data_name)
    print("num nodes: ", data.x.shape[0])
    print("node features: ", data.x.shape[1])
    print("num edges: ", data.x_1.shape[0])
    print("num triangles: ", data.x_2.shape[0])
    print("num tetrahedra: ", data.x_3.shape[0])
    print("num rank 4: ", data.x_4.shape[0])
    print("-"*50)