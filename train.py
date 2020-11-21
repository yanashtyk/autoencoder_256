# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 17:05:30 2020

@author: Administrator
"""

from model import autoencoder_256
import torch
from Images import Images
import torch.nn as nn

cuda_availability=torch.cuda.is_available()
if cuda_availability:
    device = torch.device('cuda:{}'.format(torch.cuda.current_device()))
else:
    device = 'cpu'

print (device)

org_dir = './sdo/SDOBenchmark-data-full/training'

img=Images(org_dir)

train_loader = torch.utils.data.DataLoader(img, batch_size=64 ,pin_memory=True)

model = autoencoder_256()



optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
criterion = nn.MSELoss()
num_epochs=10

model.load_state_dict(torch.load('./aut_256_20.pth'))
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
print('ok')

torch.save(model.state_dict(), './aut_256_15.pth')     

