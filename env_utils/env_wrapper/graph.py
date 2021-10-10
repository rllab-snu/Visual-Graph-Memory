import torch
class Node(object):
    def __init__(self, info=None):
        self.node_num = None
        self.time_t = None
        self.neighbors = []
        self.neighbors_node_num = []
        self.embedding = None
        self.misc_info = None
        self.action = -1
        self.visited_time = []
        self.visited_memory = []
        if info is not None:
            for k, v in info.items():
                setattr(self, k, v)

class Graph(object):
    def __init__(self, cfg, B, device):
        self.B = B
        self.memory = None
        self.memory_mask = None
        self.memory_time = None
        self.memory_num = 0
        self.input_shape = (64, 256)
        self.feature_dim = 512
        self.M = cfg.memory.memory_size
        self.torch_device = device

    def num_node(self, b):
        return len(self.node_position_list[b])

    def num_node_max(self):
        return self.graph_mask.sum(dim=1).max().long().item()

    def reset(self, B):
        if B: self.B = B
        self.node_position_list = [[] for _ in range(self.B)] # This position list is only for visualizations

        self.graph_memory = torch.zeros([self.B, self.M, self.feature_dim]).to(self.torch_device)
        self.graph_act_memory = torch.zeros([self.B, self.M],dtype=torch.uint8).to(self.torch_device)

        self.A = torch.zeros([self.B, self.M, self.M],dtype=torch.bool).to(self.torch_device)

        self.graph_mask = torch.zeros(self.B, self.M).to(self.torch_device)
        self.graph_time = torch.zeros(self.B, self.M).to(self.torch_device)

        self.last_localized_node_idx = torch.zeros([self.B], dtype=torch.int32)
        self.last_local_node_num = torch.zeros([self.B])
        self.last_localized_node_embedding = torch.zeros([self.B, self.feature_dim], dtype=torch.float32).to(self.torch_device)


    def reset_at(self,b):
        self.graph_memory[b] = 0
        self.graph_act_memory[b] = 0
        self.A[b] = 0
        self.graph_mask[b] = 0
        self.graph_time[b] = 0
        self.last_localized_node_idx[b] = 0
        self.node_position_list[b] = []

    def initialize_graph(self, b, new_embeddings, positions):
        self.add_node(b, node_idx=0, embedding=new_embeddings[b], time_step=0, position=positions[b])
        self.record_localized_state(b, node_idx=0, embedding=new_embeddings[b])

    def add_node(self, b, node_idx, embedding, time_step, position):
        self.node_position_list[b].append(position)
        self.graph_memory[b, node_idx] = embedding
        self.graph_mask[b, node_idx] = 1.0
        self.graph_time[b, node_idx] = time_step

    def record_localized_state(self, b, node_idx, embedding):
        self.last_localized_node_idx[b] = node_idx
        self.last_localized_node_embedding[b] = embedding

    def add_edge(self, b, node_idx_a, node_idx_b):
        self.A[b, node_idx_a, node_idx_b] = 1.0
        self.A[b, node_idx_b, node_idx_a] = 1.0
        return

    def update_node(self, b, node_idx, time_info, embedding=None):
        if embedding is not None:
            self.graph_memory[b, node_idx] = embedding
        self.graph_time[b, node_idx] = time_info
        return

    def update_nodes(self, bs, node_indices, time_infos, embeddings=None):
        if embeddings is not None:
            self.graph_memory[bs, node_indices] = embeddings
        self.graph_time[bs, node_indices.long()] = time_infos

    def get_positions(self, b, a=None):
        if a is None:
            return self.node_position_list[b]
        else:
            return self.node_position_list[b][a]

    def get_neighbor(self, b, node_idx, return_mask=False):
        if return_mask: return self.A[b, node_idx]
        else: return torch.where(self.A[b, node_idx])[0]

    def calculate_multihop(self, hop):
        return torch.matrix_power(self.A[:, :self.num_node_max(), :self.num_node_max()].float(), hop)

