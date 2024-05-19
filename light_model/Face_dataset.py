import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import torch.nn as nn
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()


"""
Dataset 경로
dataset
 ㄴ celeba
    ㄴ img_celeba
 ㄴ korean
    ㄴ Middle_Resolution
    ㄴ Low_Resolution
 ㄴ csvs ~~
"""
class Combined_Dataset(Dataset):
    def __init__(self, csv_file, root_dir, state, transform=None):
        """
        Args:
            csv_file (string): Path to the CSV file with labels.
            root_dir (string): Directory with all the images.
            state (string): 'train' or 'test' to filter the dataset accordingly.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.labels_frame = pd.read_csv(csv_file)
        # Filter the DataFrame based on the 'state' column
        self.labels_frame = self.labels_frame[self.labels_frame['state'] == state]
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.labels_frame.iloc[idx, 0])
        image = Image.open(img_name)
        # Assume labels are located from the second to the second-last columns
        labels = self.labels_frame.iloc[idx, 1:-1]  
        labels = torch.tensor(labels.values.astype('float'))

        if self.transform:
            image = self.transform(image)

        return image, labels
    
    

def Dataload(csv_file, img_path, transform, batch_size, state):
    
    if state =='train':
        train_dataset = Combined_Dataset(csv_file=csv_file,
                                root_dir=img_path,
                                state='train',
                                transform=transform)
        Dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True, drop_last=True, num_workers = 16)
        
    elif state == 'test':
        test_dataset = Combined_Dataset(csv_file=csv_file,
                                root_dir=img_path,
                                state='test',
                                transform=transform)
        Dataloader = DataLoader(test_dataset, batch_size = batch_size, num_workers = 16)
        
    elif state == 'valid':
        val_dataset = Combined_Dataset(csv_file=csv_file,
                                root_dir=img_path,
                                state='valid',
                                transform=transform)
        Dataloader = DataLoader(val_dataset, batch_size = batch_size, shuffle=True, drop_last=True,  num_workers = 16)    
        
    else:
        print("잘못된 값을 입력하였습니다.")
    
    return Dataloader

# if __name__ == '__main__':
    
#     transform = transforms.Compose([
#     transforms.Resize((112 , 112)),  # 이미지 크기 조정

#     transforms.ToTensor(),  # 이미지를 텐서로 변환
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 정규화
#     ])
    
#     batch_size = 243
    
#     train_path = '/hahmwj/korean_face_attribute/dataset/celeba/by_test/train/'
#     val_path = '/hahmwj/korean_face_attribute/dataset/celeba/by_test/val/'
#     test_path = '/hahmwj/korean_face_attribute/dataset/celeba/by_test/test/'

#     celebA_pd = pd.read_csv('/hahmwj/korean_face_attribute/dataset/clebA.csv')
    
#     out_test = 'Arched_Eyebrows'
#     in_test = 'Bald'
#     # print(celebA_pd.columns)

#     # for col in celebA_pd.columns:
#     #     if col != 'Young':
#     #         continue
#     # train_loader = Dataload(img_path = train_path, csv_file = celebA_pd, column_name = out_test, transform=transform, batch_size = batch_size)
#     # val_loader = Dataload(img_path = val_path, csv_file = celebA_pd, column_name = out_test, transform=transform, batch_size = batch_size)
#     test_loader = Dataload(img_path = test_path, csv_file = celebA_pd, column_name = out_test, transform=transform, batch_size = batch_size)


    
    
    
    
    
    
    
    
    