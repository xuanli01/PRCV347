import torch.nn as nn
import torch
from torch.nn.modules import linear
from transformer import TransformerBlock


class LGTRM(nn.Module):

    def __init__(self, args, hidden=768, n_layers=1, attn_heads=16, dropout=0.1):

        super().__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads
        self.linear_encoding = nn.Linear(2048, hidden)
        if args.dataset == 'shanghai':
            self.encoder_length = 112
        else:
            self.encoder_length = 405
                 
        self.pos_embedding = nn.Parameter(torch.randn(1, self.encoder_length, hidden))  
        self.dropout = nn.Dropout(dropout)
        self.feed_forward_hidden = hidden
        #DW-Net
        self.conv_1 = nn.Conv1d(in_channels=hidden, out_channels=hidden, kernel_size=3,
                      stride=1,dilation=1, padding=1,groups=hidden,bias=False) 

        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden, attn_heads, hidden, dropout) for _ in range(n_layers)])
    def forward(self, x, mask=None):

        b, n, _ = x.shape 
        #DW-Net
        x = x+self.conv_1(x.permute(0,2,1)).permute(0,2,1)

        #TF-Net
        if mask is not None:
            mask = mask.view(-1,mask.shape[-1])
            mask = torch.matmul(mask.unsqueeze(-1), mask.unsqueeze(1))
            mask = mask.unsqueeze(1).repeat(1,self.attn_heads,1,1)
            x = self.linear_encoding(x)
            x = x+self.pos_embedding[:, :n]
            x = self.dropout(x)
            for transformer in self.transformer_blocks:
                x = transformer.forward(x, mask)
        else:
            if n <= self.encoder_length:
                x = self.linear_encoding(x)
                x = x+self.pos_embedding[:, :n]
                x = self.dropout(x)
                for transformer in self.transformer_blocks:
                    x = transformer.forward(x, None)
            else:
                for i in range(n // self.encoder_length):
                    x_tmp = x[:,i*self.encoder_length:(i+1)*self.encoder_length,:]
                    x_tmp = self.linear_encoding(x_tmp)
                    x_tmp = x_tmp+self.pos_embedding
                    x_tmp = self.dropout(x_tmp)
                    for transformer in self.transformer_blocks:
                        x_tmp = transformer.forward(x_tmp, None)
                    x[:,i*self.encoder_length:(i+1)*self.encoder_length,:] = x_tmp
                if n % self.encoder_length != 0:
                    n_tmp = n % self.encoder_length 
                    x_tmp = x[:,-self.encoder_length:,:]
                    x_tmp = self.linear_encoding(x_tmp)
                    x_tmp = x_tmp+self.pos_embedding
                    x_tmp = self.dropout(x_tmp)
                    for transformer in self.transformer_blocks:
                        x_tmp = transformer.forward(x_tmp, None)
                    x[:,-n_tmp:,:] = x_tmp[:,-n_tmp:,:] 
        return x

