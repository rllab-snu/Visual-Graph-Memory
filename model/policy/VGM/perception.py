import torch
import torch.nn.functional as F
from .graph_layer import GraphConvolution
import torch.nn as nn

class Attblock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.nhead = nhead
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = F.relu
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, src, trg, src_mask):
        #q = k = self.with_pos_embed(src, pos)
        q = src.permute(1,0,2)
        k = trg.permute(1,0,2)
        src_mask = ~src_mask.bool()
        src2, attention = self.attn(q, k, value=k, key_padding_mask=src_mask)
        src2 = src2.permute(1,0,2)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, attention

class GCN(nn.Module):
    def __init__(self, input_dim, output_dim, dropout =0.1, hidden_dim=512, init='xavier'):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(input_dim, hidden_dim, init=init)
        self.gc2 = GraphConvolution(hidden_dim, hidden_dim, init=init)
        self.gc3 = GraphConvolution(hidden_dim, output_dim, init=init)
        self.dropout = nn.Dropout(dropout)

    def forward(self, batch_graph, adj):
        # make the batch into one graph
        big_graph = torch.cat([graph for graph in batch_graph],0)
        B, N = batch_graph.shape[0], batch_graph.shape[1]
        big_adj = torch.zeros(B*N, B*N).to(batch_graph.device)
        for b in range(B):
            big_adj[b*N:(b+1)*N,b*N:(b+1)*N] = adj[b]

        x = self.dropout(F.relu(self.gc1(big_graph,big_adj)))
        x = self.dropout(F.relu(self.gc2(x,big_adj)))
        big_output = self.gc3(x, big_adj)

        batch_output = torch.stack(big_output.split(N))
        return batch_output

import math
class PositionEncoding(nn.Module):
    def __init__(self, n_filters=512, max_len=2000):
        """
        :param n_filters: same with input hidden size
        :param max_len: maximum sequence length
        """
        super(PositionEncoding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, n_filters)  # (L, D)
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, n_filters, 2).float() * - (math.log(10000.0) / n_filters))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)  # buffer is a tensor, not a variable, (L, D)

    def forward(self, x, times):
        """
        :Input: (*, L, D)
        :Output: (*, L, D) the same size as input
        """
        pe = []
        for b in range(x.shape[0]):
            pe.append(self.pe.data[times[b].long()]) # (#x.size(-2), n_filters)
        pe_tensor = torch.stack(pe)
        x = x + pe_tensor
        return x

class Perception(nn.Module):
    def __init__(self,cfg, embedding_network):
        super(Perception, self).__init__()

        self.pe_method = 'pe' # or exp(-t)
        self.time_embedd_size = cfg.features.time_dim
        self.max_time_steps = cfg.TASK_CONFIG.ENVIRONMENT.MAX_EPISODE_STEPS
        self.goal_time_embedd_index = self.max_time_steps
        memory_dim = cfg.features.visual_feature_dim
        if self.pe_method == 'embedding':
            self.time_embedding = nn.Embedding(self.max_time_steps+2, self.time_embedd_size)
        elif self.pe_method == 'pe':
            self.time_embedding = PositionEncoding(memory_dim, self.max_time_steps+10)
        else:
            self.time_embedding = lambda t: torch.exp(-t.unsqueeze(-1)/5)

        feature_dim = cfg.features.visual_feature_dim# + self.time_embedd_size
        #self.feature_embedding = nn.Linear(feature_dim, memory_dim)
        self.feature_embedding = nn.Sequential(nn.Linear(feature_dim +  cfg.features.visual_feature_dim , memory_dim),
                                               nn.ReLU(),
                                               nn.Linear(memory_dim, memory_dim))

        self.global_GCN = GCN(input_dim=memory_dim, output_dim=memory_dim)
        self.goal_Decoder = Attblock(cfg.transformer.hidden_dim,
                                     cfg.transformer.nheads,
                                     cfg.transformer.dim_feedforward,
                                     cfg.transformer.dropout)
        self.curr_Decoder = Attblock(cfg.transformer.hidden_dim,
                                     cfg.transformer.nheads,
                                     cfg.transformer.dim_feedforward,
                                     cfg.transformer.dropout)

        self.output_size = feature_dim

    def normalize_sparse_adj(self, adj):
        """Laplacian Normalization"""
        rowsum = adj.sum(1) # adj B * M * M
        r_inv_sqrt = torch.pow(rowsum, -0.5)
        r_inv_sqrt[torch.where(torch.isinf(r_inv_sqrt))] = 0.
        r_mat_inv_sqrt = torch.stack([torch.diag(k) for k in r_inv_sqrt])
        return torch.matmul(torch.matmul(adj, r_mat_inv_sqrt).transpose(1,2),r_mat_inv_sqrt)

    def forward(self, observations, mode='train', return_features=False): # without memory
        B = observations['global_mask'].shape[0]
        max_node_num = observations['global_mask'].sum(dim=1).max().long()
        relative_time = observations['step'].unsqueeze(1) - observations['global_time'][:, :max_node_num]
        global_memory = self.time_embedding(observations['global_memory'][:,:max_node_num], relative_time)
        global_mask = observations['global_mask'][:,:max_node_num]
        I = torch.eye(max_node_num).unsqueeze(0).repeat(B,1,1).to(global_memory.device)
        global_A = self.normalize_sparse_adj(observations['global_A'][:,:max_node_num, :max_node_num] + I)

        goal_embedding = observations['goal_embedding']
        global_memory_with_goal= self.feature_embedding(torch.cat((global_memory[:,:max_node_num], goal_embedding.unsqueeze(1).repeat(1,max_node_num,1)),-1))
        global_context = self.global_GCN(global_memory_with_goal, global_A)

        curr_embedding, goal_embedding = observations['curr_embedding'], observations['goal_embedding']
        global_context = self.time_embedding(global_context, relative_time)
        goal_context, goal_attn = self.goal_Decoder(goal_embedding.unsqueeze(1), global_context, global_mask)
        curr_context, curr_attn = self.curr_Decoder(curr_embedding.unsqueeze(1), global_context, global_mask)

        if return_features:
            return_f = {'goal_attn': goal_attn, 'curr_attn': curr_attn}
            return curr_context.squeeze(1), goal_context.squeeze(1), return_f
        return curr_context.squeeze(1), goal_context.squeeze(1), None
