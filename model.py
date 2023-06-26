import torch
import torch.nn as nn
import torch.nn.init as torch_init
torch.set_default_tensor_type('torch.cuda.FloatTensor')
from LGTRM import LGTRM
from memory import ENPM

def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        torch_init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.batch_size =  args.batch_size
        #ENPM
        self.memory = ENPM()
        self.Aggregate = LGTRM(args, hidden=2048)
        self.fc1 = nn.Linear(args.feature_size, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 1)
        self.drop_out = nn.Dropout(0.7)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.apply(weight_init)

    def forward(self, inputs, mask):


        out = inputs
        bs, ncrops, t, f = out.size()

        out = out.view(-1, t, f)

        # LGTRM
        out = self.Aggregate(out, mask)
       
        features = self.drop_out(out)

        # classifier
        scores = self.relu(self.fc1(features))
        scores = self.drop_out(scores)
        scores = self.relu(self.fc2(scores))
        scores = self.drop_out(scores)
        scores = self.sigmoid(self.fc3(scores))
        scores = scores.view(bs, ncrops, -1).mean(1)
        scores = scores.unsqueeze(dim=2)
        if bs == 1:
            return scores
        
        normal_features = features[0:self.batch_size*10]
        mem_nor_f = out[0:self.batch_size*10]
        normal_scores = scores[0:self.batch_size]

        abnormal_features = features[self.batch_size*10:]
        mem_ab_f = out[self.batch_size*10:]
        abnormal_scores = scores[self.batch_size:]

        feat_magnitudes = torch.norm(features, p=2, dim=2)
        feat_magnitudes = feat_magnitudes.view(bs, ncrops, -1).mean(1)
        nfea_magnitudes = feat_magnitudes[0:self.batch_size]  # normal feature magnitudes
        afea_magnitudes = feat_magnitudes[self.batch_size:]  # abnormal feature magnitudes
        n_size = nfea_magnitudes.shape[0]

        
        if mask is not None:
            mask_mem = mask.clone()
            mask_mem = mask_mem.squeeze(2)
            mask = mask.mean(1).view(-1,mask.shape[-1])
            k_tot = torch.sum(mask, dim=1) // 16 + 1

        k_nor = k_tot[0:self.batch_size]
        k_abn = k_tot[self.batch_size:]
            

        select_idx = torch.ones_like(nfea_magnitudes).cuda()
        select_idx = select_idx* mask[self.batch_size:,:]
        select_idx = self.drop_out(select_idx)

        afea_magnitudes_drop = afea_magnitudes * select_idx

        mem_ab_f = mem_ab_f.view(n_size, ncrops, t, f)
        mem_ab_f = mem_ab_f.permute(1, 0, 2, 3)
        abnormal_features = abnormal_features.view(n_size, ncrops, t, f)
        abnormal_features = abnormal_features.permute(1, 0, 2, 3)

        score_abnormal = []
        total_select_abn_feature = torch.zeros(0)
        total_mem_ab_f = torch.zeros(0)
        for i in range(k_abn.shape[0]):
            k = int(k_abn[i])
            idx_abn = torch.topk(afea_magnitudes_drop[i,:], k, dim=0)[1]
            score_abnormal.append(torch.mean(abnormal_scores[i,idx_abn,:]))
            feat_select_abn = torch.norm(abnormal_features[:,i,idx_abn,:].mean(1), p=2,dim=1) 
            mask_mem[i+self.batch_size,:,:] = 0
            mask_mem[i+self.batch_size,:,idx_abn] = 1
            tmp = mem_ab_f[:,i,idx_abn,:].mean(1)
            total_mem_ab_f = torch.cat((total_mem_ab_f, tmp))
            total_select_abn_feature = torch.cat((total_select_abn_feature, feat_select_abn))

        score_abnormal = torch.stack(score_abnormal).unsqueeze(-1)    

        select_idx_normal = torch.ones_like(nfea_magnitudes).cuda()
        select_idx_normal = select_idx_normal * mask[0:self.batch_size,:]
        select_idx_normal = self.drop_out(select_idx_normal)
        nfea_magnitudes_drop = nfea_magnitudes * select_idx_normal

        mem_nor_f = mem_nor_f.view(n_size, ncrops, t, f)
        mem_nor_f = mem_nor_f.permute(1, 0, 2, 3)
        normal_features = normal_features.view(n_size, ncrops, t, f)
        normal_features = normal_features.permute(1, 0, 2, 3)

        score_normal = []
        total_select_nor_feature = torch.zeros(0)
        total_mem_nor_f = torch.zeros(0)

        for i in range(k_nor.shape[0]):
            k = int(k_nor[i])
            idx_normal = torch.topk(nfea_magnitudes_drop[i,:], k, dim=0)[1]
            score_normal.append(torch.mean(normal_scores[i,idx_normal,:]))
            feat_select_normal = torch.norm(normal_features[:,i,idx_normal,:].mean(1), p=2,dim=1) 
            mask_mem[i,:,idx_normal] = 0
            tmp = mem_nor_f[:,i,idx_normal,:].mean(1)
            total_mem_nor_f = torch.cat((total_mem_nor_f, tmp))
            total_select_nor_feature = torch.cat((total_select_nor_feature, feat_select_normal))

        score_normal = torch.stack(score_normal).unsqueeze(-1) 
        feat_select_abn = total_select_abn_feature
        feat_select_normal = total_select_nor_feature

        s_loss, c_loss,u_loss = self.memory(torch.cat((total_mem_nor_f,total_mem_ab_f),dim=0))

        return score_abnormal, score_normal, feat_select_abn, feat_select_normal, scores,s_loss,c_loss,u_loss