import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, channels):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        size = x.shape[-2:]
        x = x.flatten(2).transpose(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.transpose(1, 2).view(-1, self.channels, *size)


class UNetWithAttention(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(UNetWithAttention, self).__init__()
        
        self.time_mlp = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )
        
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels + 64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            SelfAttention(128),  
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            SelfAttention(256), 
            #nn.Conv2d(256, 512, kernel_size=3, padding=1),
            #nn.ReLU(),
            #SelfAttention(512), 
        )
        
        self.middle = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            SelfAttention(256), 
        )
        

        self.decoder = nn.Sequential(
            #nn.Conv2d(512, 256, kernel_size=3, padding=1),
            #nn.ReLU(),
            #SelfAttention(256),  
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            SelfAttention(128),  
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            SelfAttention(64),  
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1),
        )
    
    def forward(self, x, t):
        batch_size = x.shape[0]
        
        t_emb = self.time_mlp(t.view(batch_size, 1))
        t_emb = t_emb.view(batch_size, -1, 1, 1)
        t_emb = t_emb.expand(-1, -1, x.shape[2], x.shape[3])
        
        x = torch.cat([x, t_emb], dim=1)
        
        x = self.encoder(x)
        
        x = self.middle(x)
        
        x = self.decoder(x)
        
        return x
