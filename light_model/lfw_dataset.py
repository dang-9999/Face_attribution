

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import random
import os
import sys


def lfw_DataLoader(img_path, transform, batch_size, persistent_workers = False):
    #n_way, k_shot에 맞춰서 metadata 생성

        dataset = ImageFolder(root=img_path, transform = transform)
        Dataloader = DataLoader(dataset, batch_size, num_workers=16, persistent_workers=persistent_workers)
        
        return Dataloader
    


        

# def miniimagenet_MetaDataLoader(img_path, transform, n_way, k_shot, q_query ,n_episodes, persistent_workers = False):
#     # full_dataset = ImageFolder(root=img_path, transform = transform)
    
#     dataset = ImageFolder(root=img_path,  transform=transform)
    
#     labels = [label for _, label in dataset]
#     sampler = FewShotSampler(labels, n_way=n_way, k_shot=k_shot, q_query=q_query, n_episodes=n_episodes)
#     Dataloader = DataLoader(dataset, batch_sampler=sampler, num_workers=16, persistent_workers=persistent_workers)
    
#     return Dataloader
    

if __name__ == '__main__':
    import torch
    from torch.utils.data import Dataset, DataLoader
    import torch.nn as nn
    from torchvision.datasets import ImageFolder
    import torchvision.transforms as transforms
    import random
    import os
    import sys
    print("메롱")
