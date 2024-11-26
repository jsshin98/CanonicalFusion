import torch
import torch.nn as nn
import torch.nn.functional as F

class LBSUnwrappingDecoder(nn.Module):
    def __init__(self, activation = 'relu', softmax = False):
        super(LBSUnwrappingDecoder, self).__init__()
        
        act = nn.ReLU(inplace=True)
        if activation == 'leaky_relu':
            act = nn.LeakyReLU(inplace=True, negative_slope=0.1)
        elif activation == 'tanh':
            act = nn.Tanh()
        elif activation == 'prelu':
            act = nn.PReLU(init=0.25)
        elif activation == 'gelu':
            act = nn.GELU()
            
        self.softmax = softmax
        self.layer = nn.Sequential(
            nn.Linear(3, 64),
            act,
            nn.Linear(64, 128),
            act,
            nn.Linear(128, 256),
            act,
            nn.Linear(256, 128),
            act,
            nn.Linear(128, 55),
        )
        
        self.init_weights()
        
    def init_weights(self):
        # he initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                
    def forward(self, x):
        output = self.layer(x)
    
        if self.softmax:
            output = F.softmax(output, dim=1)
        
        return output

class LBSUnwrappingEncoder(nn.Module):
    def __init__(self, activation = 'relu'):
        super(LBSUnwrappingEncoder, self).__init__()
        
        act = nn.ReLU(inplace=True)
        if activation == 'leaky_relu':
            act = nn.LeakyReLU(inplace=True, negative_slope=0.1)
        elif activation == 'tanh':
            act = nn.Tanh()
        elif activation == 'prelu':
            act = nn.PReLU(init=0.25)
        elif activation == 'gelu':
            act = nn.GELU()
            
        self.sigmoid = nn.Sigmoid()
        
        self.layer = nn.Sequential(
            nn.Linear(55, 128),
            act,
            nn.Linear(128, 256),
            act,
            nn.Linear(256, 128),
            act,
            nn.Linear(128, 64),
            act,
            nn.Linear(64, 3),
        )
        
        self.init_weights()
        
    def init_weights(self):
        # he initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                
    def forward(self, x):
        output = self.layer(x)
        
        output = self.sigmoid(output)
    
        return output
    
class LBSModel(nn.Module):
    def __init__(self, activation = 'gelu', softmax = True):
        super(LBSModel, self).__init__()
        
        self.encoder = LBSUnwrappingEncoder(activation)
        self.decoder = LBSUnwrappingDecoder(activation, softmax)
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        
        return x