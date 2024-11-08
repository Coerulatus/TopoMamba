import torch

def get_bins(n_bins, incidence):
    bins = torch.zeros(n_bins+1)
    n_hyperedges = incidence.shape[1]
    ranks = torch.sum(incidence, dim=0).to_dense()
    max_rank = int(torch.max(ranks).item())
    n_ranks = torch.zeros(max_rank+1)
    for i in range(max_rank+1):
        n_ranks[i] = torch.sum(ranks == i)
    n_remaining = torch.sum(n_ranks)
    bins_remaining = n_bins
    n_per_bin = n_remaining // bins_remaining
    current_hyperedges = 0
    i = 0
    for bin in range(n_bins):
        n_per_bin = n_remaining // bins_remaining
        current_hyperedges = 0
        while current_hyperedges < n_per_bin:
            current_hyperedges += n_ranks[i]
            i += 1
        bins[bin+1] = i-1
        n_remaining -= current_hyperedges
        bins_remaining -= 1
        n_per_bin = n_remaining // bins_remaining
        current_hyperedges = 0
    return bins

def get_rank_masks(incidence, keep_empty_ranks=False, rank_limit=10000, ranks_to_keep=None):
    incidence = incidence.coalesce()
    ranks = torch.sum(incidence, dim=0)
    if ranks.is_sparse:
        ranks = ranks.to_dense()
    if keep_empty_ranks:
        if ranks_to_keep is None:
            n_ranks = min(int(torch.max(ranks).item()),rank_limit)
        else:
            n_ranks = min(ranks_to_keep.shape[0],rank_limit)
    else:
        if ranks_to_keep is None:
            ranks_to_keep = ranks.tolist()
            ranks_to_keep = set(ranks_to_keep)
            ranks_to_keep = list(ranks_to_keep)
            ranks_to_keep = sorted(ranks_to_keep, reverse=True)
            if ranks_to_keep[-1] != 1:
                ranks_to_keep.append(1)
            ranks_to_keep = torch.tensor(ranks_to_keep, device=incidence.device, dtype=torch.long)
        n_ranks = min(ranks_to_keep.shape[0],rank_limit)
    rank_masks = torch.zeros((n_ranks, incidence.shape[1]), device=incidence.device)
    for i,rank in enumerate(ranks_to_keep):
        if i==n_ranks:
            break
        rank_masks[i, ranks==rank] = 1
    return rank_masks
    
def get_binned_rank_masks(n_bins, incidence, keep_empty_ranks=False, ranks_to_keep=None):
    rank_masks = get_rank_masks(incidence, keep_empty_ranks, ranks_to_keep=ranks_to_keep)
    if n_bins >= incidence.shape[1]:
        return rank_masks
    rank_masks = torch.flip(rank_masks, [0])
    bins = get_bins(n_bins, incidence)
    binned_rank_masks = torch.zeros((n_bins, incidence.shape[1]), device=incidence.device)
    for bin in range(n_bins):
        binned_rank_masks[bin] = torch.sum(rank_masks[int(bins[bin]):int(bins[bin+1])], dim=0)
    binned_rank_masks = torch.flip(binned_rank_masks, [0])
    return torch.cat((binned_rank_masks, torch.zeros((1, incidence.shape[1]), device=incidence.device)), dim=0)

def get_distances(incidence, distance_limit):
    distances = []
    if not incidence.is_sparse:
        raise NotImplementedError("Incidence matrix must be sparse")
    d = torch.sparse.mm(incidence.t(),incidence)
    d = torch.sparse.IntTensor(d._indices(),(d._values()>0).float(),d.size())
    h_d = torch.diag(torch.ones(d.shape[0], device=incidence.device)).to_sparse()
    h_d = d.clone()-h_d
    d_old = d.clone()
    distances.append(incidence)
    running_sum = incidence.clone()
    ii = 0
    while ii<distance_limit:
        res = torch.sparse.mm(incidence, h_d)-running_sum
        distances.append(torch.sparse.FloatTensor(res._indices(), torch.clamp(res._values(),0,1), res.size()).coalesce())
        running_sum += res
        d = d.coalesce()
        d = torch.sparse.mm(d,d.t())
        d = torch.sparse.IntTensor(d._indices(),(d._values()>0).float(),d.size())
        if torch.all((d-d_old)._values()==0):
            break
        h_d = (d-d_old).clone()
        d_old = d.clone()
        ii += 1
    distances = torch.stack(distances, dim=1)
    # Reshape the distances 
    diam = distances.shape[1]
    idxs = torch.stack([distances._indices()[0,:]*diam+distances._indices()[1,:],distances._indices()[2,:]], dim=0)
    return torch.sparse.FloatTensor(idxs, distances._values(), (distances.size()[0]*distances.size()[1],distances.size()[2]))

def get_ranks_to_keep(incidence):
    ranks = torch.sum(incidence, dim=0).to_dense()
    ranks_to_keep = torch.unique(ranks)
    ranks_to_keep = sorted(ranks_to_keep, reverse=True)
    if ranks_to_keep[-1] == 0:
        ranks_to_keep.pop()
    if ranks_to_keep[-1] != 1:
        ranks_to_keep.append(1)
    ranks_to_keep = torch.tensor(ranks_to_keep, device=incidence.device).to(torch.long)
    return ranks_to_keep

if __name__ == '__main__':
    data = torch.utils.data.Dataset()
    data.x = torch.ones((4,4))
    data.x[1,:] = torch.ones(4)*2
    data.x[2,:] = torch.ones(4)*3
    data.incidence = torch.ones((4,4))
    data.incidence[2,2] = 0
    data.incidence[3,2] = 0
    data.incidence[3,3] = 0
    data.incidence = data.incidence.to_sparse()
    bins = get_bins(5, data.incidence)
    print(bins)
    rank_masks = get_rank_masks(data.incidence)
    binned_rank_masks = get_binned_rank_masks(4, data.incidence)
    print(binned_rank_masks)