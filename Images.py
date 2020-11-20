# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 16:53:28 2020

@author: Administrator
"""

from torch.utils.data import Dataset
import torchvision.transforms as transforms
import os
from PIL import Image
import torch

class Images (Dataset): #create dataset class
    
    def __init__(self, org_dir):
        self.org_dir=org_dir #in what directory are the originals
        self.transform=transforms.Compose([transforms.ToTensor()])
        self.lost=0
        self.org_numb=os.listdir(org_dir) #List of all directories numbers in originals folder
        self.org_dir_numbs=[os.path.join(org_dir, org_dir_numb) for org_dir_numb in self.org_numb ] #join folder and number folder
        self.directories=[]
        del self.org_dir_numbs[-1]
        del self.org_dir_numbs[-1]
        for i in self.org_dir_numbs:
            date=os.listdir(i)
            for d in date:
                dirct=os.path.join(i, d)
                self.directories.append(dirct)
        
        self.org_names=[]
        for i in self.directories:
            listt=os.listdir(i)
            names_1700 = [listt[i] for i in range(len(listt)) if listt[i][-8:-4]=='1700']
            names_cont=[listt[i] for i in range(len(listt)) if listt[i][-13:-4]=='continuum' ]
            names_magn=[listt[i] for i in range(len(listt)) if listt[i][-15:-4]=='magnetogram']
            
            names=[]
            for nn in names_1700:
                names_3=[]
                names_3.append(nn)
                cont=[names_cont[i] for i in range(len(names_cont)) if names_cont[i][:17]==nn[:17]]
                magn=[names_magn[i] for i in range(len(names_magn)) if names_magn[i][:17]==nn[:17]]
                if len(cont)==1 and len(magn)==1 :
                    names_3.append(cont[0])
                    names_3.append(magn[0])
                    names.append(names_3)
                else:
                    self.lost+=1
                
            for n in names:
                name=[]
                for nn in n :
                    
                    name.append(os.path.join(i, nn))
                self.org_names.append(name)
           
                             
           
        
    def __getitem__(self, index):
        res=[]
        imgg_name=self.org_names[index]
        for name in imgg_name:
            imgg=Image.open(name)
            imgg= self.transform(imgg)
            imgg=imgg.squeeze()
            res.append(imgg)
        return torch.stack(res)

    def __len__(self):
    
        return len(self.org_names)
        