from pytorch_lightning.utilities.types import EVAL_DATALOADERS
import torch
import pytorch_lightning as pl
import time
import numpy as np
import os

from topomamba.modules.sequencers import SequencerRank, SequencerOnlyX, SequencerRankConcatenate, SequencerRankBinned
from topomamba.evaluators.evaluator import Evaluator
from topomamba.modules.mlp import MLP

from topomamba.models.s4 import S4Model
# from topomamba.modules.gru import GRUCell

from mamba_ssm import Mamba

# from s4torch import S4Model

class TopoMamba(pl.LightningModule):
    def __init__(self,
                 d_input,
                 d_out,
                 d_hidden = 256,
                 n_layers = 2,
                 sequencer = 'rank',
                 sequencer_limit = 10000,
                 l_max = 2,
                 n_bins = 64,
                 keep_empty_ranks = True,
                 same_graph = False,
                 task_level = "node",
                 task = "classification",
                 pooling_type = "sum",
                 loss = torch.nn.CrossEntropyLoss(),
                 lr = 0.01,
                 weight_decay = 0.0,
                 input_dropout = 0.1,
                 readout_dropout = 0.1,
                 skip_connection = False,
                 backwards = True,
                 device = "cuda",
                 log = True,
                 feature_lifting = "sum",
                 aggregation = "sum",
                 sequence_model = 'mamba',
                 batch_size = 1,
                 save_results = False,
                 time_start = time.time()):
        super(TopoMamba, self).__init__()
        self.loss = loss
        self.lr = lr
        
        self.input_transform = torch.nn.Sequential(torch.nn.Linear(d_input, d_hidden, device=device),
                                                    torch.nn.Dropout(input_dropout),
                                                    torch.nn.ReLU(),)
        if sequencer=='rank':
            self.sequencer = SequencerRank(rank_limit=sequencer_limit, 
                                           same_graph=same_graph,
                                           keep_empty_ranks=keep_empty_ranks,
                                           feature_lifting=feature_lifting,
                                           d_hidden=d_hidden)
        elif sequencer=='only_x':
            self.sequencer = SequencerOnlyX()
        elif sequencer=='rank_concatenate':
            self.sequencer = SequencerRankConcatenate(rank_limit=sequencer_limit,
                                                      same_graph=same_graph,
                                                      keep_empty_ranks=keep_empty_ranks)
        elif sequencer=='rank_binned':
            self.sequencer = SequencerRankBinned(n_bins=n_bins,
                                                 same_graph=same_graph,
                                                 keep_empty_ranks=keep_empty_ranks)
        else:
            raise NotImplementedError("Sequencer {} not implemented".format(sequencer))
        self.layer_norm = torch.nn.LayerNorm(d_hidden, device=device)
        if sequence_model == 'mamba':
            self.layers = torch.nn.ModuleList([Mamba(d_model=d_hidden, d_state=16, d_conv=4, expand=2, device=device) for _ in range(n_layers)])
            self.layers_back = torch.nn.ModuleList([Mamba(d_model=d_hidden, d_state=16, d_conv=4, expand=2, device=device) for _ in range(n_layers)])
        elif sequence_model == 'gru':
            self.layers = torch.nn.ModuleList([torch.nn.GRU(input_size=d_hidden, hidden_size=d_hidden, batch_first=True, device=device) for _ in range(n_layers)])
            self.layers_back = torch.nn.ModuleList([torch.nn.GRU(input_size=d_hidden, hidden_size=d_hidden, batch_first=True, device=device) for _ in range(n_layers)])
        elif sequence_model == 'lstm':
            self.layers = torch.nn.ModuleList([torch.nn.LSTM(input_size=d_hidden, hidden_size=d_hidden, batch_first=True, device=device) for _ in range(n_layers)])
            self.layers_back = torch.nn.ModuleList([torch.nn.LSTM(input_size=d_hidden, hidden_size=d_hidden, batch_first=True, device=device) for _ in range(n_layers)])
        elif sequence_model == 's4':
            self.layers = torch.nn.ModuleList([S4Model(d_hidden, d_hidden, d_hidden, n_layers=2, dropout=0.2) for _ in range(n_layers)])
            self.layers_back = torch.nn.ModuleList([S4Model(d_hidden, d_hidden, d_hidden, n_layers=2, dropout=0.2) for _ in range(n_layers)])
            self.layers = torch.nn.ModuleList([layer.to(device) for layer in self.layers])
            self.layers_back = torch.nn.ModuleList([layer.to(device) for layer in self.layers_back])
        elif sequence_model == 'transformer':
            self.layers = torch.nn.ModuleList([torch.nn.TransformerEncoderLayer(d_hidden, nhead=16, dim_feedforward=d_hidden, device=device, batch_first=True) for _ in range(n_layers)])
            self.layers_back = torch.nn.ModuleList([torch.nn.TransformerEncoderLayer(d_hidden, nhead=16, dim_feedforward=d_hidden, device=device, batch_first=True) for _ in range(n_layers)])
        else:
            raise NotImplementedError("Sequence model {} not implemented".format(sequence_model))
        
        self.readout = MLP(in_channels=d_hidden,
                           hidden_channels=d_hidden,
                           out_channels=d_out,
                           num_layers=1,
                           dropout=readout_dropout,
                           Normalization='bn',
                           InputNorm=False)
        self.task_level = task_level
        self.task = task
        self.pooling_type = pooling_type
        self.evaluator = Evaluator(task=task)
        self.train_results = {"logits": [], "labels": []}
        self.val_results = {"logits": [], "labels": []}
        self.test_results = {"logits": [], "labels": []}
        self.do_log = log
        self.skip_connection = skip_connection
        self.do_backwards = backwards
        self.sequencer_limit = sequencer_limit
        self.sequencer_name = sequencer
        self.feature_lifting = feature_lifting
        self.aggregation = aggregation
        self.batch_size = batch_size
        self.sequence_model = sequence_model
        self.save_hyperparameters(ignore=['loss'])
        self.train_times = []
        self.train_loss = []
        self.val_acc = []
        self.train_iter = []
        self.train_epoch = []
        self.val_times = []
        self.n_epoch = 1
        self.time = time_start
        self.save_results = save_results
        self.weight_decay = weight_decay
        if self.save_results:
            self.idx = 0
            # check if the file already exists
            while os.path.exists(f'./results/train_loss{self.idx}.npy'):
                self.idx += 1
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        self.params = sum([np.prod(p.size()) for p in model_parameters])
        
    def forward(self, batch):
        n_nodes = batch.x.shape[0]
        if hasattr(batch, 'incidence'):
            incidence = batch.incidence#_batch
        else:
            incidence = torch.sparse_coo_tensor(
                torch.stack((batch.edge_index0, batch.edge_index1)),
                values=torch.ones(batch.edge_index0.shape[0], device=batch.x.device),
                size=(n_nodes, torch.sum(batch.num_hyperedges).item()),
            ).coalesce()
        x = self.input_transform(batch.x)
        for layer, layer_back in zip(self.layers, self.layers_back):
            if self.skip_connection:
                residual = x
            x = self.sequencer(x, incidence, batch.he_id)
            x = self.layer_norm(x)
            if self.sequence_model == 'mamba' or self.sequence_model == 'gru' or self.sequence_model == 'lstm' or self.sequence_model == 's4' or self.sequence_model == 'transformer':
                if self.do_backwards:
                    x_back =  torch.flip(x, dims=[1])
                if self.sequence_model == 'mamba' or self.sequence_model == 's4':
                    x = layer(x)
                    if self.do_backwards:
                        x_back = layer_back(x_back)
                elif self.sequence_model == 'gru' or self.sequence_model == 'lstm':
                    x = layer(x)[0]
                    if self.do_backwards:
                        x_back = layer_back(x_back)[0]
                x_back = torch.flip(x, dims=[1])
                if self.do_backwards:
                    x = x + x_back
            if self.aggregation == "sum":
                x = torch.sum(x, dim=1)
            elif self.aggregation == "mean":
                x = torch.mean(x, dim=1)
            elif self.aggregation == "max":
                x = torch.max(x, dim=1)[0]
            if self.skip_connection:
                x += residual
        if self.task_level == "graph":
            if self.pooling_type == "sum":
                x = torch.sum(x, dim=0)
            elif self.pooling_type == "mean":
                x = torch.mean(x, dim=0)
            elif self.pooling_type == "max":
                x = torch.max(x, dim=0)
            else:
                raise NotImplementedError
        x = self.readout(x)
        return x
    
    def balance_loss(self, y):
        if max(y) == 1:
            num_ones = torch.sum(y)
            num_zeros = len(y) - num_ones
            self.loss = torch.nn.CrossEntropyLoss(weight=torch.tensor([1/num_zeros, 1/num_ones]).float().to(y.device))
        else:
            raise NotImplementedError("Balance loss only implemented for binary classification")
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer

    def training_step(self, batch, batch_idx):
        if self.task == "classification":
            mask = torch.tensor([i for i in range(batch.batch_size)])
        else:
            mask = torch.tensor([i for i in range(batch.y.shape[0])])
        y_true = batch.y[mask]
        mask = mask.to('cpu')
        y_pred = self.forward(batch)
        mask = mask.to(batch.x.device)
        y_pred = y_pred[mask]
        if self.task == "classification":
            loss = self.loss(y_pred, torch.nn.functional.one_hot(y_true, num_classes=y_pred.shape[1]).float())
        else:
            loss = self.loss(y_pred.squeeze(), y_true)
        if self.do_log:
            if self.batch_size == 'full':
                batch_size = len(y_true)
            else:
                batch_size = self.batch_size
            self.log("train_loss", loss, batch_size=batch_size)
        self.train_results["logits"].append(y_pred.detach())
        self.train_results["labels"].append(y_true)
        if hasattr(self, 'time_start_train'):
            self.train_iter.append(time.time()-self.time_start_train)
        self.time_start_train = time.time()
        self.train_times.append(time.time()-self.time)
        self.train_loss.append(loss.cpu().detach().numpy())
        return loss
    
    def on_train_epoch_end(self):
        if len(self.train_results["logits"][0].shape) == 0:
            y_pred = torch.stack(self.train_results["logits"],dim=0)
            y_true = torch.stack(self.train_results["labels"],dim=0)
        else:
            y_pred = torch.cat(self.train_results["logits"],dim=0)
            y_true = torch.cat(self.train_results["labels"],dim=0)
        eval = self.evaluator.eval({"logits": y_pred.squeeze(), "labels": y_true})
        for key in eval:
            if self.do_log:
                if self.batch_size == 'full':
                    batch_size = len(y_true)
                else:
                    batch_size = self.batch_size
                self.log("train_"+key, eval[key], batch_size=batch_size)
                self.log("a_train_iter", np.mean(self.train_iter))
                self.log("a_train_iter_std", np.std(self.train_iter))
                self.log("a_train_epoch", np.mean(self.train_epoch))
                self.log("a_train_epoch_std", np.std(self.train_epoch))        
                if torch.cuda.is_available():
                    gpu_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # in MB
                    torch.cuda.reset_peak_memory_stats()
                    self.log("train_gpu_mem", gpu_memory, batch_size=batch_size)
                    self.log("_n_params", self.params)
        self.train_results = {"logits": [], "labels": []}
        if hasattr(self, 't_epoch'):
            self.train_epoch.append(time.time()-self.t_epoch)
        self.t_epoch = time.time()

    def validation_step(self, batch, batch_idx):
        time_start = time.time()
        if self.task == "classification":
            mask = torch.tensor([i for i in range(batch.batch_size)])
        else:
            mask = torch.tensor([i for i in range(batch.y.shape[0])])
        y_true = batch.y[mask]
        y_pred = self.forward(batch)
        y_pred = y_pred[mask]
        if self.task == "classification":
            loss = self.loss(y_pred, torch.nn.functional.one_hot(y_true, num_classes=y_pred.shape[1]).float())
        else:
            loss = self.loss(y_pred.squeeze(), y_true)
        if self.do_log:
            if self.batch_size == 'full':
                batch_size = len(y_true)
            else:
                batch_size = self.batch_size
            self.log("valid_loss", loss, batch_size=batch_size)
        self.val_results["logits"].append(y_pred.detach())
        self.val_results["labels"].append(y_true)
        return loss
    
    def on_validation_epoch_end(self):
        if len(self.val_results["logits"][0].shape) == 0:
            y_pred = torch.stack(self.val_results["logits"],dim=0)
            y_true = torch.stack(self.val_results["labels"],dim=0)
        else:
            y_pred = torch.cat(self.val_results["logits"],dim=0)
            y_true = torch.cat(self.val_results["labels"],dim=0)
        eval = self.evaluator.eval({"logits": y_pred.squeeze(), "labels": y_true})
        for key in eval:
            if self.batch_size == 'full':
                batch_size = len(y_true)
            else:
                batch_size = self.batch_size
            if key == 'acc':
                self.log("valid_acc", eval["acc"], prog_bar=True, batch_size=batch_size)
            elif self.do_log:
                self.log("valid_"+key, eval[key], batch_size=batch_size)
                self.n_epoch += 1
            self.val_times.append(time.time()-self.time)
            if hasattr(eval, 'acc'):
                self.val_acc.append(eval["acc"])
        self.val_results = {"logits": [], "labels": []}
    
    def on_train_end(self) -> None:
        if self.save_results:
            idx = 0
            # check if the file already exists
            while os.path.exists(f'./results/train_loss{idx}.npy'):
                idx += 1
            # save train_loss and train_time with numpy
            with open(f'./results/train_loss{idx}.npy', 'wb') as f:
                np.save(f, self.train_loss)
            with open(f'./results/train_time{idx}.npy', 'wb') as f:
                np.save(f, self.train_times)
            with open(f'./results/val_acc{idx}.npy', 'wb') as f:
                np.save(f, self.val_acc)
            with open(f'./results/val_time{idx}.npy', 'wb') as f:
                np.save(f, self.val_times)
        return super().on_train_end()

    def test_step(self, batch, batch_idx):
        if self.task == "classification":
            mask = torch.tensor([i for i in range(batch.batch_size)])
        else:
            mask = torch.tensor([i for i in range(batch.y.shape[0])])
        y_true = batch.y[mask]
        y_pred = self.forward(batch)
        y_pred = y_pred[mask]
        self.test_results["logits"].append(y_pred.detach())
        self.test_results["labels"].append(y_true)
        
    def on_test_epoch_end(self):
        if len(self.test_results["logits"][0].shape) == 0:
            y_pred = torch.stack(self.test_results["logits"],dim=0)
            y_true = torch.stack(self.test_results["labels"],dim=0)
        else:
            y_pred = torch.cat(self.test_results["logits"],dim=0)
            y_true = torch.cat(self.test_results["labels"],dim=0)
        eval = self.evaluator.eval({"logits": y_pred.squeeze(), "labels": y_true})
        for key in eval:
            if self.batch_size == 'full':
                batch_size = len(y_true)
            else:
                batch_size = self.batch_size
            if self.do_log:
                self.log("test_"+key, eval[key], batch_size=batch_size)
        self.test_results = {"logits": [], "labels": []}
        
        return eval
        