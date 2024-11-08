import random
import torch
import time
try:
    from topomamba.modules.utils import get_rank_masks, get_binned_rank_masks, get_distances, get_ranks_to_keep
except:
    from utils import get_rank_masks, get_binned_rank_masks, get_distances, get_ranks_to_keep
    

class SequencerRank(torch.nn.Module):
    def __init__(self, rank_limit=10000, same_graph=False, keep_empty_ranks=True, feature_lifting="sum", d_hidden=0, return_ranks=False):
        super().__init__()
        self.same_graph = same_graph
        self.rank_limit = rank_limit
        self.rank_masks = None
        self.keep_empty_ranks = keep_empty_ranks
        self.ranks_to_keep = None
        self.feature_lifting = feature_lifting
        self.return_ranks = return_ranks

    def setup(self, incidence):
        self.ranks_to_keep = get_ranks_to_keep(incidence)
        self.rank_masks = get_rank_masks(incidence, self.keep_empty_ranks, self.rank_limit, self.ranks_to_keep)
    
    def forward(self, x, incidence, he_idxs):
        self.rank_masks = self.rank_masks.to(incidence.device)
        if self.feature_lifting == "attention":
            incidence = incidence.coalesce()
            edge_index_h = incidence.indices()
            hyperedge_features = self.pma_hyperedges(x, edge_index_h)
            ranks_features = torch.einsum('ij,jk->jik', self.rank_masks, hyperedge_features)
            ranks_features_flat = ranks_features.reshape(ranks_features.shape[0],-1)
            edge_index_h = incidence.coalesce().T.coalesce().indices()
            sequences = self.pma_ranks(ranks_features_flat, edge_index_h).to_dense()
        elif self.feature_lifting == "mean":
            rank_masks = self.rank_masks[:, he_idxs]
            hyperedge_features = torch.sparse.mm(torch.transpose(incidence,1,0), x)
            ranks = torch.sum(incidence, dim=0).values()
            hyperedge_features = torch.div(hyperedge_features, ranks.unsqueeze(1))            
            ranks_features = torch.einsum('ij,jk->jik', rank_masks, hyperedge_features)
            sequences = torch.sparse.mm(incidence, ranks_features.reshape(ranks_features.shape[0],-1)).to_dense()
        else:
            rank_masks = self.rank_masks[:, he_idxs]
            hyperedge_features = torch.sparse.mm(torch.transpose(incidence,1,0), x)
            ranks_features = torch.einsum('ij,jk->jik', rank_masks, hyperedge_features)
            sequences = torch.sparse.mm(incidence, ranks_features.reshape(ranks_features.shape[0],-1)).to_dense()
        sequences = sequences.reshape(x.shape[0],-1,ranks_features.shape[2])
        sequences[:,-1,:] = x
        if self.return_ranks:
            return sequences, hyperedge_features, rank_masks
        return sequences

class SequencerOnlyX(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def setup(self, incidence):
        pass
    
    def forward(self, x, incidence, he_idxs=None):
        return torch.unsqueeze(x, dim=1)

# class SequencerRankConcatenate(torch.nn.Module):
#     def __init__(self, rank_limit=10000, same_graph=False, keep_empty_ranks=False, return_idxs=False):
#         super().__init__()
#         self.same_graph = same_graph
#         self.rank_limit = rank_limit
#         self.rank_masks = None
#         self.max_degree = None
#         self.keep_empty_ranks = keep_empty_ranks
#         self.ranks_to_keep = None
#         self.return_idxs = return_idxs
        
#     def setup(self, incidence):
#         self.ranks_to_keep = get_ranks_to_keep(incidence)
    
#     def forward(self, x, incidence, node2hedgePE=None):
#         if self.rank_masks is None or not self.same_graph:
#             self.rank_masks = get_rank_masks(incidence, self.keep_empty_ranks, self.rank_limit, self.ranks_to_keep)
#             row_sum = torch.sum(incidence, dim=1).to_dense()
#             col_sum = torch.sum(incidence, dim=0).to_dense()
#             self.nodes_to_assign = [i for i,row_s in enumerate(row_sum) for _ in range(int(row_s))]
#             self.n_hyperedges = [i+1 for row_s in row_sum for i in range(int(row_s))]
#             #TODO: remove this argsort and do it only in the for loop
#             rank_order = torch.argsort(col_sum)
#             index_map = {i.item(): idx for idx,i in enumerate(rank_order)}
#             self.hyperedges_to_assign = []
#             self.sorted_hedge_idxs = []
#             current_n_hedges = 0
#             for n in range(len(row_sum)):
#                 hedge_idxs = incidence._indices().t()[incidence._indices().t()[:,0] == n][:,1]
#                 rank_sorted_hedge_idxs = hedge_idxs[torch.argsort(torch.tensor([index_map[idx.item()] for idx in hedge_idxs]))]
#                 self.hyperedges_to_assign.append(rank_sorted_hedge_idxs)
#                 self.sorted_hedge_idxs.append(torch.sort(rank_sorted_hedge_idxs)[1]+current_n_hedges)
#                 current_n_hedges += len(rank_sorted_hedge_idxs)
#             self.hyperedges_to_assign = torch.cat(self.hyperedges_to_assign)
#             self.sorted_hedge_idxs = torch.cat(self.sorted_hedge_idxs)
#             # n_hyperedges_per_rank = torch.sparse.mm(incidence,self.rank_masks.t())
#             # idxs_to_shuffle = [i*self.rank_masks.shape[0]+j for i,row in enumerate(n_hyperedges_per_rank) for j,r in enumerate(row) for _ in range(int(r.item()))]
#             # self.indices_dict = {}
#             # for idx, value in enumerate(idxs_to_shuffle):
#             #     if value not in self.indices_dict:
#             #         self.indices_dict[value] = []
#             #     self.indices_dict[value].append(idx)
#             self.max_degree = int(torch.max(torch.sum(incidence, dim=1).to_dense()).item())
#         # for value_indices in self.indices_dict.values():
#         #     random.shuffle(value_indices)
#         # shuffled_indices = [idx for indices in self.indices_dict.values() for idx in indices]
#         # hyperedges_to_assign = self.hyperedges_to_assign[idxs_to_shuffle]
#         hyperedge_features = torch.sparse.mm(torch.transpose(incidence,1,0), x)
#         additional_dim = 0
#         hidden_dim = x.shape[1]
#         if node2hedgePE is not None:
#             additional_dim = 0#node2hedgePE.shape[-1]
#         sequences = torch.zeros((x.shape[0],self.max_degree+1,hidden_dim+additional_dim), device=x.device)
#         sequences[self.nodes_to_assign,self.n_hyperedges,:hidden_dim] = hyperedge_features[self.hyperedges_to_assign,:]
#         if node2hedgePE is not None:
#             pe = torch.zeros((sequences.shape[0], sequences.shape[1], node2hedgePE.shape[-1]), device=x.device)
#             pe[self.nodes_to_assign,self.n_hyperedges,:] = node2hedgePE[self.nodes_to_assign, torch.tensor(self.n_hyperedges, device=self.sorted_hedge_idxs.device)[self.sorted_hedge_idxs]-1]
#         sequences[:,-1,:hidden_dim] = x
#         if self.return_idxs and node2hedgePE is not None:
#             return sequences, torch.tensor(self.nodes_to_assign), self.n_hyperedges, self.hyperedges_to_assign, self.sorted_hedge_idxs, pe 
#         elif self.return_idxs:
#             return sequences, torch.tensor(self.nodes_to_assign), self.n_hyperedges, self.hyperedges_to_assign, self.sorted_hedge_idxs
#         return sequences

# class SequencerRankBinned(torch.nn.Module):
#     def __init__(self, n_bins, same_graph=False, keep_empty_ranks=True):
#         super().__init__()
#         self.n_bins = n_bins
#         self.same_graph = same_graph
#         self.rank_masks = None
#         self.keep_empty_ranks = keep_empty_ranks
#         self.ranks_to_keep = None
    
#     def setup(self, incidence):
#         self.ranks_to_keep = get_ranks_to_keep(incidence)
#         self.rank_masks = get_binned_rank_masks(self.n_bins, incidence, self.keep_empty_ranks, self.ranks_to_keep)
        
#     def forward(self, x, incidence, he_idxs):
#         self.rank_masks = self.rank_masks.to(incidence.device)
#         rank_masks = self.rank_masks[:, he_idxs]
#         hyperedge_features = torch.sparse.mm(torch.transpose(incidence,1,0), x)
#         ranks_features = torch.einsum('ij,jk->jik', rank_masks, hyperedge_features)
        
#         sequences = torch.sparse.mm(incidence, ranks_features.reshape(ranks_features.shape[0],-1)).to_dense()
#         sequences = sequences.reshape(sequences.shape[0],-1,ranks_features.shape[2])
#         sequences[:,-1,:] = x
#         return sequences
           
if __name__ == '__main__':
    sequencer1 = SequencerRank(keep_empty_ranks=False)
    data1 = torch.utils.data.Dataset()
    data1.x = torch.ones((3,4))
    data1.x[1,:] = torch.ones(4)*2
    data1.x[2,:] = torch.ones(4)*3
    data1.incidence = torch.ones((3,2))
    data1.incidence[2,0] = 0
    data1.incidence = data1.incidence.to_sparse()
    result1 = sequencer1(data1.x, data1.incidence)
    assert result1.shape == (3,3,4)
    assert torch.all(result1[0,:,:] == torch.tensor([[6,6,6,6],[3,3,3,3],[1,1,1,1]]))
    assert torch.all(result1[1,:,:] == torch.tensor([[6,6,6,6],[3,3,3,3],[2,2,2,2]]))
    assert torch.all(result1[2,:,:] == torch.tensor([[6,6,6,6],[0,0,0,0],[3,3,3,3]]))
    
    sequencer2 = SequencerOnlyX()
    result2 = sequencer2(data1.x, data1.incidence)
    assert result2.shape == (3,1,4)
    assert torch.all(result2[0,:,:] == torch.tensor([[1,1,1,1]]))
    assert torch.all(result2[1,:,:] == torch.tensor([[2,2,2,2]]))
    assert torch.all(result2[2,:,:] == torch.tensor([[3,3,3,3]]))
     
    # sequencer3 = SequencerRankConcatenate()
    # data2 = torch.utils.data.Dataset()
    # data2.x = torch.ones((4,4))
    # data2.x[1,:] = torch.ones(4)*2
    # data2.x[2,:] = torch.ones(4)*3
    # data2.x[3,:] = torch.ones(4)*4
    # data2.incidence = torch.ones((4,3))
    # data2.incidence[2,1] = 0
    # data2.incidence[1,2] = 0
    # data2.incidence[3,0] = 0
    # data2.incidence[3,2] = 0
    # data2.incidence = data2.incidence.to_sparse()
    # result3 = sequencer3(data2.x, data2.incidence)
    # assert result3.shape == (4,4,4)
    # assert torch.all(result3[0,:,:] == torch.tensor([[7,7,7,7],[6,6,6,6],[4,4,4,4],[1,1,1,1]]))
    # assert torch.all(result3[1,:,:] == torch.tensor([[0,0,0,0],[7,7,7,7],[6,6,6,6],[2,2,2,2]]))
    # assert torch.all(result3[2,:,:] == torch.tensor([[0,0,0,0],[6,6,6,6],[4,4,4,4],[3,3,3,3]]))
    # assert torch.all(result3[3,:,:] == torch.tensor([[0,0,0,0],[0,0,0,0],[7,7,7,7],[4,4,4,4]]))
    
    # sequencer4 = SequencerRankBinned(n_bins=1)
    # result4 = sequencer4(data1.x, data1.incidence)
    # assert result4.shape == (3,2,4)
    # assert torch.all(result4[0,:,:] == torch.tensor([[9,9,9,9],[1,1,1,1]]))
    # assert torch.all(result4[1,:,:] == torch.tensor([[9,9,9,9],[2,2,2,2]]))
    # assert torch.all(result4[2,:,:] == torch.tensor([[6,6,6,6],[3,3,3,3]]))
    
    print("All tests passed")