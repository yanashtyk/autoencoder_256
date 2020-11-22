# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 11:07:42 2020

@author: Administrator
"""

import torch.nn as nn
import torch.nn.functional as F

class aut_256_conv(nn.Module):
    def __init__(self):
        super(aut_256_conv, self).__init__()
        
        #image size 256*256
        
        self.encoder=nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=5),
            nn.ReLU(True),
            nn.Conv2d(64, 192, kernel_size=5, stride=3),
            nn.ReLU(True),
            nn.Conv2d(192, 384, kernel_size=5, stride=1),
            nn.ReLU(True),
            nn.Conv2d(384, 256, kernel_size=3, stride=3),
            nn.ReLU(True),
            nn.Conv2d(256, 50, kernel_size=4, stride=1),
            nn.ReLU(True))
        
        self.decoder=nn.Sequential(
            
            nn.ConvTranspose2d(50, 256, kernel_size=4, stride=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 384, kernel_size=3, stride=3),
            nn.ReLU(True),
            nn.ConvTranspose2d(384, 192, kernel_size=5, stride=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(192, 64, kernel_size=5, stride=3),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, kernel_size=11, stride=5),
            nn.ReLU(True))
            
            
        
        
        
    def forward(self, x):
            
        x=self.encoder(x)
        print(x.shape)
        x=self.decoder(x)
        return x
        
        
        
        
        