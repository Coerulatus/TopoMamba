import torch
import pytorch_lightning as pl
from torch_geometric.loader import NeighborLoader, DataLoader

from topomamba.data.utils import get_one_incidence, reduce_incidence, clean_data

from torch_geometric.loader import NeighborLoader
from torch_geometric.loader.utils import filter_data, filter_node_store_

# Custom index_select function to avoid in-place updates
def custom_index_select(value, index, dim=0):
    return torch.index_select(value, dim, index).clone()

# Custom filter_node_store_ function to avoid in-place updates
def custom_filter_node_store_(store, out_store, node):
    for key, value in store.items():
        if key in out_store:
            out_store[key] = custom_index_select(value, node, dim=0)
        else:
            out_store[key] = custom_index_select(value, node, dim=0)

# Custom NeighborLoader class
class CustomNeighborLoader(NeighborLoader):
    def filter_fn(self, out):
        data = out.data
        store = data._store
        node = out.node
        out_store = data.__class__()

        custom_filter_node_store_(store, out_store, node)
        
        # Set the filtered store back to data
        data._store = out_store
        return data
    
class PLDatamoduleBatch(pl.LightningDataModule):
    def __init__(self, dataset, batch_size=64, max_n_neighbors=100000, n_hops=1, model='mamba'):
        super().__init__()
        self.dataset = dataset
        self.max_n_neighbors = max_n_neighbors
        self.n_hops = n_hops
        self.batch_size = batch_size
        self.model = model
        self.setup()
        
    def setup(self):
        if self.model == 'mamba':
            self.dataset = get_one_incidence(self.dataset)
        self.train_idxs = self.dataset.train_mask
        self.val_idxs = self.dataset.val_mask
        self.test_idxs = self.dataset.test_mask
        self.dataset = clean_data(self.dataset)
        
    def train_dataloader(self):
        if self.batch_size == 'full':
            return NeighborLoader(self.dataset, num_neighbors=[self.max_n_neighbors for _ in range(self.n_hops)], input_nodes=self.train_idxs,  batch_size=len(self.train_idxs), shuffle=True, transform=reduce_incidence)
        return NeighborLoader(self.dataset, num_neighbors=[self.max_n_neighbors for _ in range(self.n_hops)], input_nodes=self.train_idxs,  batch_size=self.batch_size, shuffle=True, transform=reduce_incidence)#, num_workers=self.num_workers)

    def val_dataloader(self):
        if self.batch_size == 'full':
            return NeighborLoader(self.dataset, num_neighbors=[self.max_n_neighbors for _ in range(self.n_hops)], input_nodes=self.val_idxs,  batch_size=len(self.val_idxs), shuffle=False, transform=reduce_incidence)
        return NeighborLoader(self.dataset, num_neighbors=[self.max_n_neighbors for _ in range(self.n_hops)], input_nodes=self.val_idxs,  batch_size=self.batch_size, shuffle=False, transform=reduce_incidence)
    
    def test_dataloader(self):
        if self.batch_size == 'full':
            return NeighborLoader(self.dataset, num_neighbors=[self.max_n_neighbors for _ in range(self.n_hops)], input_nodes=self.test_idxs,  batch_size=len(self.test_idxs), shuffle=False, transform=reduce_incidence)
        return NeighborLoader(self.dataset, num_neighbors=[self.max_n_neighbors for _ in range(self.n_hops)], input_nodes=self.test_idxs,  batch_size=self.batch_size, shuffle=False, transform=reduce_incidence)
    