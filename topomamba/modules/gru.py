import torch
from torch.nn import functional as F

class GRUCell(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, process_whole_sequence, device) -> None:
        super(GRUCell, self).__init__()
        self.input_dim, self.hidden_dim = input_dim, hidden_dim
        self.relevance_whh, self.relevance_wxh, self.relevance_b = self.create_gate_parameters(device)
        self.update_whh, self.update_wxh, self.update_b = self.create_gate_parameters(device)
        self.candidate_whh, self.candidate_wxh, self.candidate_b = self.create_gate_parameters(device)
        self.process_whole_sequence = process_whole_sequence
        self.attention = torch.nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, device=device, batch_first=True)
        self.num_heads = 4

    def create_gate_parameters(self, device):
        input_weights = torch.nn.Parameter(torch.zeros(self.input_dim, self.hidden_dim))
        hidden_weights = torch.nn.Parameter(torch.zeros(self.hidden_dim, self.hidden_dim))
        torch.nn.init.xavier_uniform_(input_weights)
        torch.nn.init.xavier_uniform_(hidden_weights)
        bias = torch.nn.Parameter(torch.zeros(self.hidden_dim))
        return hidden_weights.to(device), input_weights.to(device), bias.to(device)
    
    def forward(self, x, x_rank, incidence, he_features, rank_mask, batch_size, h=None):
        if h is None:
            h = torch.zeros(x.shape[0], self.hidden_dim, device=x.device)
        if self.process_whole_sequence:
            output_hiddens = []
            for i in range(x_rank.shape[1]):
                relevance_gate = F.sigmoid((h @ self.relevance_whh) + (x_rank[:, i] @ self.relevance_wxh) + self.relevance_b)
                update_gate = F.sigmoid((h @ self.update_whh) + (x_rank[:, i] @ self.update_wxh) + self.update_b)
                candidate_hidden = F.tanh(((relevance_gate * h) @ self.candidate_whh) + (x_rank[:, i] @ self.candidate_wxh) + self.candidate_b)
                h = (update_gate * candidate_hidden) + ((1 - update_gate) * h)
                output_hiddens.append(h.unsqueeze(1))
            output_hiddens = torch.concat(output_hiddens, dim=1)
            
            results = []
            he_order = []
            for i in range(batch_size):
                he_idxs_x_i = []
                indices = incidence.coalesce().indices()
                he_x_i = indices[1,torch.where(indices[0]==i)[0]]
                max_l = 0
                for j in range(rank_mask.shape[0]):
                    he_x_i_rank_j = he_x_i[torch.where(rank_mask[j,he_x_i]==1)[0]]
                    he_idxs_x_i.append(he_x_i_rank_j.tolist())
                    if len(he_idxs_x_i[-1])>max_l:
                        max_l = len(he_idxs_x_i[-1])
                d_h = he_features.shape[-1]
                he_features_x_i = [[h for h in he_features[he_idxs_x_i[j]].tolist()]+[[0 for _ in range(d_h)]]*(max_l-len(he_idxs_x_i[j])) for j in range(len(he_idxs_x_i))]
                mask = [[[1] for _ in he_features[he_idxs_x_i[j]].tolist()]+[[0]]*(max_l-len(he_idxs_x_i[j])) for j in range(len(he_idxs_x_i))]
                he_features_x_i = torch.tensor(he_features_x_i, device=x.device)
                mask = torch.tensor(mask, device=x.device)
                mask = mask.repeat(self.num_heads, 1, 1).to(torch.bool)
                
                k = torch.permute(output_hiddens[i].unsqueeze(0), (1, 0, 2))
                q = he_features_x_i
                if len(q.shape)!=2:
                    out = self.attention(q,k,k)[0]#,attn_mask=mask)[0]
                    results.append(out[mask[:out.shape[0],:,0]])
                    he_order.append([h for list in he_idxs_x_i for h in list])
                else:
                    print(i)
            current_n = 0
            he_final_order = []
            for h in he_order:
                he_final_order.append(torch.argsort(torch.tensor(h))+current_n)
                current_n += len(h)
                # ranks = []
                # for hh in h:
                #     ranks.append(len(indices[0,indices[1]==hh]))
                # assert torch.all(torch.tensor(ranks) == torch.sort(torch.tensor(ranks), descending=True)[0])
            return torch.cat(results, dim=0), torch.tensor([h.item() for t in he_final_order for h in t])
        else:
            raise NotImplementedError("Only processing whole sequences is supported")