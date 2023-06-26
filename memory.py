from scipy import sparse
import torch
import torch.nn as nn
import math
from torch.nn import functional as F


torch.set_default_tensor_type('torch.cuda.FloatTensor')
class ENPM(nn.Module):
    def __init__(self, NBlock=2000, dim=2048):
        super(ENPM, self).__init__()
        self.NBlock = NBlock
        self.dim = dim
        self.weight = nn.Parameter(torch.randn(NBlock, dim))
        self.bias = None
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
    
    def forward(self,x, mask=None):

        if mask is not None:
            smask = mask
            diversity_loss = torch.norm(F.linear(self.weight, self.weight) - torch.eye(self.NBlock),p='fro')
            att_weight = F.linear(F.normalize(x,p=2,dim=2), F.normalize(self.weight,p=2,dim=1)) 
            tmp = att_weight[0:x.shape[0] // 2,:,:].permute(2,0,1) * smask[0:x.shape[0] // 2,:]
            tmp = F.softmax(torch.sum(tmp,dim=-1),dim=0)
            tmp = torch.mean(tmp,dim=1)
            u_loss = torch.norm(tmp,p=2)
            max_att_weight, index = torch.topk(att_weight,2,dim=2)
            min_sim = []
            ab_min = []
            for i in range(x.shape[0] // 2):
                idx = torch.nonzero(smask[i,:]).squeeze(-1)
                min_sim.append(torch.min(max_att_weight[i,idx,0]))
                ab_idx = torch.nonzero(smask[i+(x.shape[0] // 2),:]).squeeze(-1)
                ab_min.append(torch.min(max_att_weight[i+(x.shape[0] // 2),ab_idx,0]))
            min_sim = torch.stack(min_sim)
            ab_min = torch.stack(ab_min)
            return 0.2*diversity_loss, (2-torch.mean(min_sim) +torch.mean(ab_min)),0.5*u_loss

        else:
            att_weight = F.linear(F.normalize(x,p=2,dim=1), F.normalize(self.weight,p=2,dim=1)) 
            max_att_weight, index = torch.topk(att_weight,2,dim=1)
            tmp = F.softmax(att_weight[0:x.shape[0] // 2,:],dim=1)
            tmp = torch.mean(tmp,dim=0)
            u_loss = torch.norm(tmp,p=2)
            margin = max_att_weight[0:x.shape[0] // 2,0] - max_att_weight[0:x.shape[0] // 2,1]
            min_sim = max_att_weight[0:x.shape[0] // 2,0]
            ab_min = max_att_weight[x.shape[0] // 2:,0]

            return 2-torch.mean(margin), 2-torch.mean(min_sim) +torch.mean(ab_min),0.5*u_loss
