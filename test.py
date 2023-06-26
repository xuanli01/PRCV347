import torch
from sklearn.metrics import auc, roc_curve
import numpy as np
from torch.utils.data import DataLoader
from dataset import Dataset
import option
from model import Model
import os

def test(dataloader, model, args, device):
    with torch.no_grad():
        model.eval()
        pred = torch.zeros(0)

        for i, input in enumerate(dataloader):
            input = input.to(device)
            input = input.permute(0, 2, 1, 3)
            logits = model(input, None)
            logits = torch.squeeze(logits, 1)
            logits = torch.mean(logits, 0)
            sig = logits
            pred = torch.cat((pred, sig))

        if args.dataset == 'shanghai':
            gt = np.load('list/gt-sh.npy')
        else:
            gt = np.load('list/gt-ucf.npy')

        pred = list(pred.cpu().detach().numpy())
        pred = np.repeat(np.array(pred), 16)

        fpr, tpr, threshold = roc_curve(list(gt), pred)
        rec_auc = auc(fpr, tpr)
        print('auc : ' + str(rec_auc))
        return rec_auc

def make_testing_list(args):
    dataset = args.dataset
    data_root = args.data_root
    if dataset == 'shanghai':
        res = open('list/shanghai-i3d-test.list','w')
        with open('list/sh.list', 'r') as f:
            for lines in f.readlines():
                res.write(os.path.join(data_root, lines.split('/')[-1]))
    else: 
        
        res = open('list/ucf-i3d-test.list','w')
        with open('list/ucf.list', 'r') as f:
            for lines in f.readlines():
                res.write(os.path.join(data_root, lines.split('/')[-1]))   


if __name__ == '__main__':
    args = option.parser.parse_args()
    make_testing_list(args)
    test_loader = DataLoader(Dataset(args, test_mode=True),
                              batch_size=1, shuffle=False,
                              num_workers=0, pin_memory=False)
    
    model = Model(args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.dataset != 'shanghai':
        model = torch.nn.DataParallel(model, device_ids=[0])
    
    model = model.to(device)   
    model.load_state_dict(torch.load(args.checkpoint))
    auc = test(test_loader, model, args, device)
