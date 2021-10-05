import torch

class BatchQueue(object):
    def __init__(self, B, max_len=40):
        self.B = B
        self.data = torch.ones([B, max_len]) * -1
        self.num = torch.zeros(B, dtype=torch.int64)
    def put(self, b, data):
        new_len = len(data)
        self.data[b,self.num[b]:self.num[b]+new_len] = torch.tensor(data)
        self.num[b] += new_len

    def pop_b(self, b):
        ret_vector = self.data[b,0]
        self.data[b] = torch.cat((self.data[b,1:], torch.ones(len(b),1)*-1),1)
        self.num[b] -= 1
        return ret_vector

    def pop(self, batch_mode=False):
        if not batch_mode or (self.num <= 0).all():
            ret_vector = self.data[:,0]
            self.data = torch.cat((self.data[:,1:], torch.ones(self.B,1)*-1),1)
            self.num = torch.clamp(self.num-1,0)
            return ret_vector
        else:
            not_zero = torch.where(self.num > 0)[0]
            min_num = self.num[not_zero].min()
            ret_vector = self.data[:, :min_num]
            self.data = torch.cat((self.data[:,min_num:], torch.ones(self.B,min_num)*-1),1)
            self.num = torch.clamp(self.num-min_num,0)
            return ret_vector # shape B * min_num

    def clear_b(self,b):
        self.data[b,:] = -1
        self.num[b] = 0

    def clear(self):
        self.data[:,:] = -1
        self.num[:] = 0
class Node(object):
    def __init__(self, info=None):
        # basic elements in single node
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

