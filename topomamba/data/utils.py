import torch
import torch_sparse
import torch_geometric

class DotDict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    
def reduce_incidence(batch):
    """
    Reduce the incidence matrix to only include the edges that are present in the batch.
    """
    if hasattr(batch, 'incidence'):
        batch.incidence = batch.incidence.coalesce()
    if hasattr(batch, 'incidence_1'):
        batch.incidence_1 = batch.incidence_1.coalesce()
    if hasattr(batch, 'incidence_2'):
        batch.incidence_2 = batch.incidence_2.coalesce()
    if hasattr(batch, 'laplacian_0'):
        batch.laplacian_0 = batch.laplacian_0.coalesce()
    if hasattr(batch, 'laplacian_1'):
        batch.laplacian_1 = batch.laplacian_1.coalesce()
    if hasattr(batch, 'laplacian_2'):
        batch.laplacian_2 = batch.laplacian_2.coalesce()
        
    if hasattr(batch, 'incidence'):
        sparse_tensor = torch_sparse.SparseTensor(row=batch.incidence.indices()[0],
                                                col=batch.incidence.indices()[1],
                                                value=torch.ones(batch.incidence.indices().shape[1]),
                                                sparse_sizes=batch.incidence.size())
        idxs = torch.where(torch_sparse.sum(sparse_tensor, dim=0)>0)
        reduced_incidence = torch_sparse.index_select(sparse_tensor, dim=1, idx=idxs[0])
        # shuffled_columns = torch.randperm(reduced_incidence.size(1))
        # reduced_incidence = reduced_incidence[:,shuffled_columns]
        batch.he_id = idxs[0]#[shuffled_columns]
        # batch.shuffle_idxs = shuffled_columns
        batch.incidence = reduced_incidence.to_torch_sparse_coo_tensor().to(batch.incidence.device)
    if hasattr(batch, 'incidence_1'):
        sparse_incidence_1 = torch_sparse.SparseTensor(row=batch.incidence_1.indices()[0],
                                                        col=batch.incidence_1.indices()[1],
                                                        value=torch.ones(batch.incidence_1.indices().shape[1]),
                                                        sparse_sizes=batch.incidence_1.size())
        idxs_edges = torch.where(torch_sparse.sum(sparse_incidence_1, dim=0)>0)
        reduced_incidence_1 = torch_sparse.index_select(sparse_incidence_1, dim=1, idx=idxs_edges[0])
        batch.incidence_1 = reduced_incidence_1.to_torch_sparse_coo_tensor().to(batch.incidence_1.device)
    else: 
        return batch
    if hasattr(batch, 'incidence_2'):
        sparse_incidence_2 = torch_sparse.SparseTensor(row=batch.incidence_2.indices()[0],
                                                        col=batch.incidence_2.indices()[1],
                                                        value=torch.ones(batch.incidence_2.indices().shape[1]),
                                                        sparse_sizes=batch.incidence_2.size())
        reduced_incidence_2 = torch_sparse.index_select(sparse_incidence_2, dim=0, idx=idxs_edges[0])
        idxs_triangles = torch.where(torch_sparse.sum(reduced_incidence_2, dim=0)>0)
        reduced_incidence_2 = torch_sparse.index_select(reduced_incidence_2, dim=1, idx=idxs_triangles[0])
        batch.incidence_2 = reduced_incidence_2.to_torch_sparse_coo_tensor().to(batch.incidence_2.device)
    if hasattr(batch, 'laplacian_0'):
        sparse_laplacian_0 = torch_sparse.SparseTensor(row=batch.laplacian_0.indices()[0],
                                                        col=batch.laplacian_0.indices()[1],
                                                        value=batch.laplacian_0.values(),
                                                        sparse_sizes=batch.laplacian_0.size())
        node_idxs = batch.n_id
        reduced_laplacian_0 = torch_sparse.index_select(sparse_laplacian_0, dim=1, idx=node_idxs)
        batch.laplacian_0 = reduced_laplacian_0.to_torch_sparse_coo_tensor().to(batch.laplacian_0.device)
    if hasattr(batch, 'laplacian_1'):
        sparse_laplacian_1 = torch_sparse.SparseTensor(row=batch.laplacian_1.indices()[0],
                                                        col=batch.laplacian_1.indices()[1],
                                                        value=batch.laplacian_1.values(),
                                                        sparse_sizes=batch.laplacian_1.size())
        reduced_laplacian_1 = torch_sparse.index_select(sparse_laplacian_1, dim=0, idx=idxs_edges[0])
        reduced_laplacian_1 = torch_sparse.index_select(reduced_laplacian_1, dim=1, idx=idxs_edges[0])
        batch.laplacian_1 = reduced_laplacian_1.to_torch_sparse_coo_tensor().to(batch.laplacian_1.device)
    if hasattr(batch, 'laplacian_2'):
        sparse_laplacian_2 = torch_sparse.SparseTensor(row=batch.laplacian_2.indices()[0],
                                                        col=batch.laplacian_2.indices()[1],
                                                        value=batch.laplacian_2.values(),
                                                        sparse_sizes=batch.laplacian_2.size())
        reduced_laplacian_2 = torch_sparse.index_select(sparse_laplacian_2, dim=0, idx=idxs_triangles[0])
        reduced_laplacian_2 = torch_sparse.index_select(reduced_laplacian_2, dim=1, idx=idxs_triangles[0])
        batch.laplacian_2 = reduced_laplacian_2.to_torch_sparse_coo_tensor().to(batch.laplacian_2.device)
    if hasattr(batch, 'down_laplacian_1'):
        sparse_down_laplacian_1 = torch_sparse.SparseTensor(row=batch.down_laplacian_1.indices()[0],
                                                        col=batch.down_laplacian_1.indices()[1],
                                                        value=batch.down_laplacian_1.values(),
                                                        sparse_sizes=batch.down_laplacian_1.size())
        reduced_down_laplacian_1 = torch_sparse.index_select(sparse_down_laplacian_1, dim=0, idx=idxs_edges[0])
        reduced_down_laplacian_1 = torch_sparse.index_select(reduced_down_laplacian_1, dim=1, idx=idxs_edges[0])
        batch.down_laplacian_1 = reduced_down_laplacian_1.to_torch_sparse_coo_tensor().to(batch.down_laplacian_1.device)
    if hasattr(batch, 'up_laplacian_1'):
        sparse_up_laplacian_1 = torch_sparse.SparseTensor(row=batch.up_laplacian_1.indices()[0],
                                                        col=batch.up_laplacian_1.indices()[1],
                                                        value=batch.up_laplacian_1.values(),
                                                        sparse_sizes=batch.up_laplacian_1.size())
        reduced_up_laplacian_1 = torch_sparse.index_select(sparse_up_laplacian_1, dim=0, idx=idxs_edges[0])
        reduced_up_laplacian_1 = torch_sparse.index_select(reduced_up_laplacian_1, dim=1, idx=idxs_edges[0])
        batch.up_laplacian_1 = reduced_up_laplacian_1.to_torch_sparse_coo_tensor().to(batch.up_laplacian_1.device)
    if hasattr(batch, 'down_laplacian_2'):
        batch.down_laplacian_2 = batch.down_laplacian_2.coalesce()
        sparse_down_laplacian_2 = torch_sparse.SparseTensor(row=batch.down_laplacian_2.indices()[0],
                                                        col=batch.down_laplacian_2.indices()[1],
                                                        value=batch.down_laplacian_2.values(),
                                                        sparse_sizes=batch.down_laplacian_2.size())
        reduced_down_laplacian_2 = torch_sparse.index_select(sparse_down_laplacian_2, dim=0, idx=idxs_triangles[0])
        reduced_down_laplacian_2 = torch_sparse.index_select(reduced_down_laplacian_2, dim=1, idx=idxs_triangles[0])
        batch.down_laplacian_2 = reduced_down_laplacian_2.to_torch_sparse_coo_tensor().to(batch.down_laplacian_2.device)
    if hasattr(batch, 'up_laplacian_2'):
        sparse_up_laplacian_2 = torch_sparse.SparseTensor(row=batch.up_laplacian_2.indices()[0],
                                                        col=batch.up_laplacian_2.indices()[1],
                                                        value=batch.up_laplacian_2.values(),
                                                        sparse_sizes=batch.up_laplacian_2.size())
        reduced_up_laplacian_2 = torch_sparse.index_select(sparse_up_laplacian_2, dim=0, idx=idxs_triangles[0])
        reduced_up_laplacian_2 = torch_sparse.index_select(reduced_up_laplacian_2, dim=1, idx=idxs_triangles[0])
        batch.up_laplacian_2 = reduced_up_laplacian_2.to_torch_sparse_coo_tensor().to(batch.up_laplacian_2.device)
        
    return batch

def get_one_incidence(data):
    complex = 1
    indices = torch.zeros((2,0))
    complex2nodes = {}
    num_complexes = 0
    while True:
        try:
            incidence_i = data[f'incidence_{complex}']
        except:
            break
        if complex == 1:
            indices = incidence_i.indices()
        else:
            edges = {}
            for i in range(incidence_i.shape[1]):
                edges[i] = torch.unique(incidence_i.indices()[0,incidence_i.indices()[1]==i]).tolist()
                for j in range(complex-1,0,-1):
                        edges[i] = torch.unique(torch.tensor([n for c in edges[i] for n in complex2nodes[j][c]])).tolist()
            try:
                new_indices = torch.tensor([[n, e+num_complexes] for e, key in enumerate(edges) for n in edges[key]]).permute(1,0)
            except:
                pass
            indices = torch.cat((indices, new_indices), dim=1)
        
        num_complexes += incidence_i.shape[1]
        complex2nodes[complex] = {}
        for i in range(incidence_i.shape[1]):
            complex2nodes[complex][i] = torch.unique(incidence_i.indices()[0,incidence_i.indices()[1]==i]).tolist()
        complex += 1
    values = torch.ones(indices.shape[1])
    size = (data.incidence_1.shape[0], num_complexes)
    incidence = torch.sparse_coo_tensor(indices=indices, values=values, size=size, device=data.incidence_1.device).coalesce()
    data.incidence = incidence
    for i in range(complex):
        if hasattr(data, f'incidence_{i}'):
            del data[f'incidence_{i}']
        if hasattr(data, f'laplacian_{i}'): 
            del data[f'laplacian_{i-1}']
        if hasattr(data, f'up_laplacian_{i}'):
            del data[f'up_laplacian_{i-1}']
        if hasattr(data, f'down_laplacian_{i}'):
            del data[f'down_laplacian_{i-1}']
    del data.incidence_1
    return data

def clean_data(data):
    new_data = torch_geometric.data.Data()
    new_data.x = data.x
    new_data.edge_index = data.edge_index
    new_data.y = data.y
    if hasattr(data,'incidence'):
        new_data.incidence = data.incidence
    if hasattr(data,'hodge_laplacian_0'):
        new_data.laplacian_0 = data.hodge_laplacian_0
    if hasattr(data,'hodge_laplacian_1'):
        new_data.laplacian_1 = data.hodge_laplacian_1
    if hasattr(data,'hodge_laplacian_2'):
        new_data.laplacian_2 = data.hodge_laplacian_2
    if hasattr(data,'incidence_1'):
        new_data.incidence_1 = data.incidence_1
    if hasattr(data,'incidence_2'):
        new_data.incidence_2 = data.incidence_2
    if hasattr(data,'down_laplacian_1'):
        new_data.down_laplacian_1 = data.down_laplacian_1
    if hasattr(data,'up_laplacian_1'):
        new_data.up_laplacian_1 = data.up_laplacian_1
    if hasattr(data,'down_laplacian_2'):
        new_data.down_laplacian_2 = data.down_laplacian_2
    if hasattr(data,'up_laplacian_2'):
        new_data.up_laplacian_2 = data.up_laplacian_2
    return new_data
