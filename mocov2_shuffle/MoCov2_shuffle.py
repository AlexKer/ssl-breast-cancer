import numpy as np
import torchio as tio
import torch
import torch.nn as nn
import torch.nn.functional as F
from resnet import resnet18, resnet34, resnet50

########################################################## Model ##########################################################
class MoCov2(nn.Module):
    def __init__(self, dim=128, resnet_type=50, K=68, m=0.999):
        super(MoCov2, self).__init__()
        if resnet_type == 18:
            self.encoder_q = resnet18()
            self.encoder_k = resnet18()
        elif resnet_type == 34:
            self.encoder_q = resnet34()
            self.encoder_k = resnet34()
        elif resnet_type == 50:
            self.encoder_q = resnet50()
            self.encoder_k = resnet50()

        self.K = K
        self.m = m
        self.T = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        reduce_out_features = self.encoder_q.reduce[2].out_features
        original_reduce_q = self.encoder_q.reduce[:4]
        self.encoder_q.reduce = nn.Sequential(
            original_reduce_q,
            nn.Linear(reduce_out_features, reduce_out_features),
            nn.ReLU(),
            nn.Linear(reduce_out_features, dim)
        )

        reduce_out_features = self.encoder_k.reduce[2].out_features
        original_reduce_k = self.encoder_k.reduce[:4]
        self.encoder_k.reduce = nn.Sequential(
            original_reduce_k,
            nn.Linear(reduce_out_features, reduce_out_features),
            nn.ReLU(),
            nn.Linear(reduce_out_features, dim)
        )

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        self.register_buffer("queue", torch.randn(dim * 4, K))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
            ):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
    
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        keys = concat_all_gather(keys)
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K
        self.queue_ptr[0] = ptr

    def forward(self, im_q_0, im_q_1, im_q_2, im_q_3,
                im_k_0, im_k_1, im_k_2, im_k_3):
        # q_1, q_2, q_3, q_4: (N, 60, 230, 230)
        q_1 = self.encoder_q(im_q_0).squeeze() # （N, dim）
        q_2 = self.encoder_q(im_q_1).squeeze() # （N, dim）
        q_3 = self.encoder_q(im_q_2).squeeze() # （N, dim）
        q_4 = self.encoder_q(im_q_3).squeeze() # （N, dim）

        with torch.no_grad():
            self._momentum_update_key_encoder()
            # k_1, k_2, k_3, k_4: (N, 60, 230, 230)
            k_1 = self.encoder_k(im_k_0).squeeze() # （N, dim）
            k_2 = self.encoder_k(im_k_1).squeeze() # （N, dim）
            k_3 = self.encoder_k(im_k_2).squeeze() # （N, dim）
            k_4 = self.encoder_k(im_k_3).squeeze() # （N, dim）

        if len(q_1.shape) == 1:
            q_1 = q_1.unsqueeze(0)
            q_2 = q_2.unsqueeze(0)
            q_3 = q_3.unsqueeze(0)
            q_4 = q_4.unsqueeze(0)
        if len(k_1.shape) == 1:
            k_1 = k_1.unsqueeze(0)
            k_2 = k_2.unsqueeze(0)
            k_3 = k_3.unsqueeze(0)
            k_4 = k_4.unsqueeze(0)
        
        q = torch.cat([q_1, q_2, q_3, q_4], dim=1) # （N, dim * 4）
        q = F.normalize(q, dim=1)
        k = torch.cat([k_1, k_2, k_3, k_4], dim=1) # （N, dim * 4）
        k = F.normalize(k, dim=1)

        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1) # （N, 1）
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()]) # （N, K）

        logits = torch.cat([l_pos, l_neg], dim=1) # （N, K+1）
        logits /= self.T

        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device) # （N, ）
        self._dequeue_and_enqueue(k)

        return logits, labels
        
@torch.no_grad()
def concat_all_gather(tensor):
    if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
        tensors_gather = [torch.ones_like(tensor)
                           for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
        output = torch.cat(tensors_gather, dim=0)
    else:
        output = tensor
    return output

########################################################## Shuffle ##########################################################
def kendall_tau_distance(uniq_num):
    original = np.array([0, 1, 2, 3])
    seen = set()
    unique_lis = []

    while len(unique_lis) < uniq_num:
        perm = np.random.permutation(original)
        if tuple(perm) not in seen:
            seen.add(tuple(perm))
            unique_lis.append(perm)
    
    dis = []
    for perm in unique_lis:
        index_map = {element: i for i, element in enumerate(perm)}
        kendall_dis = 0
        n = len(original)
        for i in range(n):
            for j in range(i+1, n):
                if index_map[original[i]] > index_map[original[j]]:
                    kendall_dis += 1
        dis.append(kendall_dis)
    
    max_dis = max(dis)
    max_index = dis.index(max_dis)
    return unique_lis[max_index]


########################################################## Data Argumentation ##########################################################
class MoCov2DataAugmentation(nn.Module):
    def __init__(self):
        super(MoCov2DataAugmentation, self).__init__()
        self.transform = tio.Compose([
          tio.RandomFlip(axes=(0, 1, 2), flip_probability=0.5),
          tio.RandomAffine(scales=(0.9, 1.1), degrees=10),
          tio.RandomBlur(std=(0.1, 2.0)),
          tio.CropOrPad((80, 204, 204)),  # if needed
          tio.ZNormalization()
      ])

    def forward(self, x):
        q = self.transform(x)
        k = self.transform(x)
        return q, k
    

    

        