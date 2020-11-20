# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import torch.nn as nn
import torch.nn.functional as F

class autoencoder_256(nn.Module):
    def __init__(self):
        super(autoencoder_256, self).__init__()
        
        #image size 256*256
        
        self.conv1=nn.Conv2d(3, 64, kernel_size=11, stride=5)
        self.conv2=nn.Conv2d(64, 192, kernel_size=4, stride=1)
        self.conv3=nn.Conv2d(192, 384, kernel_size=3, stride=1)
        self.conv4=nn.Conv2d(384, 256, kernel_size=4, stride=1)
        self.conv5=nn.Conv2d(256, 100, kernel_size=3, stride=1)
        
        self.pool1=nn.MaxPool2d(kernel_size=2, stride=2, padding=0, return_indices=True)
        self.pool2=nn.MaxPool2d(kernel_size=2, stride=2, padding=0, return_indices=True)
        self.pool3=nn.MaxPool2d(kernel_size=2, stride=2, padding=0, return_indices=True)
        
        self.unpool3=nn.MaxUnpool2d(kernel_size=2, stride=2, padding=0)
        self.unpool2=nn.MaxUnpool2d(kernel_size=2, stride=2, padding=0)
        self.unpool1=nn.MaxUnpool2d(kernel_size=2, stride=2, padding=0)
        
        self.deconv5=nn.ConvTranspose2d(100, 256, kernel_size=3, stride=1)
        self.deconv4=nn.ConvTranspose2d(256, 384, kernel_size=4, stride=1)
        self.deconv3=nn.ConvTranspose2d(384, 192, kernel_size=3, stride=1)
        self.deconv2=nn.ConvTranspose2d(192, 64, kernel_size=4, stride=1)
        self.deconv1=nn.ConvTranspose2d(64, 3, kernel_size=11, stride=5)
        
        
    def forward(self, x):
            
        
        
        x=self.conv1(x)
        x=F.relu(x)
        size_1=x.size()
        x, idxs1=self.pool1(x)
        
        x=self.conv2(x)
        x=F.relu(x)
        
        size_2=x.size()
        x, idxs2=self.pool2(x)
        
       
        x=self.conv3(x)
        x=F.relu(x)
        x=self.conv4(x)
        x=F.relu(x)
        size_3=x.size()
        x, idxs3=self.pool3(x)
        
       
        x=self.conv5(x)
        x=F.relu(x)
        
        
        #decoder
        
        x=self.deconv5(x)
        x=F.relu(x)
        x=self.unpool3(x, idxs3)
        
        x=self.deconv4(x)
        x=F.relu(x)
        x=self.deconv3(x)
        
        
        x=self.unpool2(x, idxs2)
        x=F.relu(x)
        x=self.deconv2(x)
        
        x=self.unpool1(x, idxs1)
        x=F.relu(x)
        x=self.deconv1(x)
        
        return x
        
        
        