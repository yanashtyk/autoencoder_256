# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 13:29:29 2020

@author: Administrator
"""

from model_256_conv import aut_256_conv
import torch
from Images import Images
import torch.nn as nn
cuda_availability=torch.cuda.is_available()
if cuda_availability:
    device = torch.device('cuda:{}'.format(torch.cuda.current_device()))
else:
    device = 'cpu'

print (device)

org_dir = './SDOBenchmark-data-all/training'

img=Images(org_dir)

train_loader = torch.utils.data.DataLoader(img, batch_size=132 ,pin_memory=True)

model = aut_256_conv()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
criterion = nn.MSELoss()
num_epochs=10
print('load')

for epoch in range(num_epochs):
    for data in train_loader:
        model.train()
        im = data
        im.to(device)
        output = model(im)
        loss = criterion(output, im)
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # ===================loss========================
    print('epoch [{}/{}], loss:{:.4f}'
          .format(epoch+1, num_epochs, loss.item()))
    torch.save(model.state_dict(), '/home/yanashtykk/aut_conv_{}'.format(epoch+1))
print('ok')

torch.save(model.state_dict(), './aut_conv_10.pth')     
