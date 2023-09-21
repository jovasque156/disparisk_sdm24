# From https://github.com/UCSB-NLP-Chang/Fairness-Reprogramming/blob/master/CelebA/dataset.py

import os
import numpy as np
import torch
import h5py
import pandas
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor

import ipdb

attr_list = ('5_o_Clock_Shadow,Arched_Eyebrows,Attractive,Bags_Under_Eyes,Bald,Bangs,Big_Lips,Big_Nose,'
             'Black_Hair,Blond_Hair,Blurry,Brown_Hair,Bushy_Eyebrows,Chubby,Double_Chin,Eyeglasses,Goatee,Gray_Hair,'
             'Heavy_Makeup,High_Cheekbones,Male,Mouth_Slightly_Open,Mustache,Narrow_Eyes,No_Beard,Oval_Face,Pale_Skin,'
             'Pointy_Nose,Receding_Hairline,Rosy_Cheeks,Sideburns,Smiling,Straight_Hair,Wavy_Hair,Wearing_Earrings,'
             'Wearing_Hat,Wearing_Lipstick,Wearing_Necklace,Wearing_Necktie,Young'
             ).split(',')

class CelebSpu(Dataset):
    def __init__(self, root, transform=ToTensor(), type="train") -> None:
        super().__init__()
        self.type = type
        self.img_dir = os.path.join(root, 'img_align_celeba_png')
        self.table = self.__load_table(os.path.join(root, 'list_attr_celeba.csv'))
        self.transform = transform
        self.num_classes = 4

    def __len__(self):
        return len(self.table)

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_dir, self.table.iloc[index, 0]))
        if self.transform is not None:
            img = self.transform(img)
        label = self.table.iloc[index]['Male'] * 2 + self.table.iloc[index]['Black_Hair']
        return img, label

    def __load_table(self, path):
        whole_table = pandas.read_csv(path)
        train_point = 162770
        val_point = 182637
        if self.type == "train":
            return whole_table.iloc[:train_point]
        elif self.type == "val":
            return whole_table.iloc[train_point:val_point]
        elif self.type == "test":
            return whole_table.iloc[val_point:]
        else:
            raise ValueError("Invalid dataset type!")


class CelebA(Dataset):
    def __init__(self, root_dir, target_attrs, domain_attrs=None, img_transform=ToTensor(), type="train") -> None:
        super().__init__()
        self.type = type
        self.img_dir = os.path.join(root_dir, 'img_align_celeba_png')
        self.table = self.__load_table(os.path.join(root_dir, 'list_attr_celeba.csv'))
        self.target_attrs = target_attrs
        self.domain_attrs = domain_attrs
        self.img_transform = img_transform

    def __len__(self):
        return len(self.table)

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_dir, self.table.iloc[index, 0]))
        if self.img_transform is not None:
            img = self.img_transform(img)
        labels = self.table[self.target_attrs].iloc[index] if isinstance(self.target_attrs, str) else \
            self.table[self.target_attrs].iloc[index].to_numpy()
        if self.domain_attrs is not None:
            domains = self.table[self.domain_attrs].iloc[index] if isinstance(self.domain_attrs, str) else \
                self.table[self.domain_attrs].iloc[index].to_numpy()
            return img, (labels, domains)
        else:
            return img, labels

    def __load_table(self, path):
        whole_table = pandas.read_csv(path)
        train_point = 162770
        val_point = 182637
        if self.type == "train":
            return whole_table.iloc[:train_point]
        elif self.type == "val":
            return whole_table.iloc[train_point:val_point]
        elif self.type == "test":
            return whole_table.iloc[val_point:]
        else:
            raise ValueError("Invalid dataset type!")

class CelebAFast(Dataset):
    def __init__(self, root, target_attrs, domain_attrs=None, img_transform=None, land_marks=None, keep_land_marks=False, size_mask=20, type="train", root_landmarks='data/list_landmarks_align_celeba.csv') -> None:
        super().__init__()
        assert type in ["train", "val", "test"]
        self.type = type
        # self.true_type only in  ["train", "val", "test"]
        self.true_type = type if type != "trigger" else "train"
        self.keep_land_marks = keep_land_marks
        self.root = root
        self.img_transform = img_transform
        self.size_mask = size_mask
        if isinstance(target_attrs, str):
            self.target_attrs = [bytes(target_attrs, 'utf-8')]
        else:
            self.target_attrs = [bytes(target_attr, 'utf-8') for target_attr in target_attrs]
        
        if domain_attrs is not None:
            if isinstance(domain_attrs, str):
                self.domain_attrs = [bytes(domain_attrs, 'utf-8')]
            else:
                self.domain_attrs = [bytes(domain_attr, 'utf-8') for domain_attr in domain_attrs]
        else:
            self.domain_attrs = None

        if isinstance(target_attrs, list):
            self.num_classes = 2 ** len(self.target_attrs)
        else:
            self.num_classes = 2

        if land_marks is not None:
            if isinstance(land_marks, str):
                self.land_marks = [land_marks]
            else:
                self.land_marks = land_marks
            self.landmarks_table = self.__load_table(root_landmarks)
        else:
            self.land_marks = None
        
        self.labels = []
        self.landmarks = []
        self.y_index = [] #y is the target
        self.z_index = [] #z is the sensitive attribute
        self.landmarks_index = [] # landmarks for masking the image before providing it

        with h5py.File(self.root, mode='r') as file:
            # ipdb.set_trace()
            if isinstance(np.array(file["columns"])[0], str):
                # Sometimes np.array(file["columns"])[0] is bytes and sometimes it's string for different systems,
                # so when it is a string we need to change target_attrs back to string
                self.target_attrs = target_attrs if isinstance(target_attrs, list) else [target_attrs]
                if domain_attrs is not None:
                    self.domain_attrs = domain_attrs if isinstance(domain_attrs, list) else [domain_attrs]
            
            self.y_index = [np.where(np.array(file["columns"]) == target_attr)[0][0] 
                            for target_attr
                            in self.target_attrs]
            
            if self.domain_attrs is not None:
                self.z_index = [np.where(np.array(file["columns"]) == domain_attr)[0][0] 
                                for domain_attr
                                in self.domain_attrs]
            
            if self.land_marks is not None:
                self.landmarks_index = [np.where(np.array(self.landmarks_table.columns) == land_mark)[0][0]
                                        for land_mark
                                        in self.land_marks]
            
            self.labels = []
            self.total = file[self.true_type]['label'].shape[0]
            if type == "train":
                self.start_point = 0
                self.end_point = self.total
            else:
                self.start_point = 0
                self.end_point = self.total
            
            # Labels are the annotated attribtues
            for i in tqdm(range(self.start_point, self.end_point)):
                self.labels.append(file[self.true_type]['label'][i])
                if self.land_marks is not None: self.landmarks.append(self.landmarks_table.iloc[i].to_numpy())
            
            self.lens = len(self.labels)
            self.y_statistics = [0,1] # [mean, std]
            self.z_statistics = [0,1] # [mean, std]

    def __load_table(self, path):
        whole_table = pandas.read_csv(path)
        train_point = 162770
        val_point = 182637
        if self.type == "train":
            return whole_table.iloc[:train_point]
        elif self.type == "val":
            return whole_table.iloc[train_point:val_point]
        elif self.type == "test":
            return whole_table.iloc[val_point:]
        else:
            raise ValueError("Invalid dataset type!")

    def __len__(self):
        return self.lens

    def __getitem__(self, index):
        # Do not open the file in the __init__, this will disable the num-workers.
        with h5py.File(self.root, mode='r') as file:
            # This is designed for "train" and "trigger", they share file["train"] but different start_point.
            # For "val" and "test", self.start_point + index = indx
            image = torch.Tensor(file[self.true_type]['data'][self.start_point + index] / 255.).permute(2, 0, 1)
            
            labels = self.get_label(index)
            
            if self.land_marks is not None:
                
                mask = torch.ones_like(image, dtype=torch.bool) if self.keep_land_marks else torch.zeros_like(image, dtype=torch.bool)
                    
                land_marks = labels[-1]
                
                for i in range(0,len(land_marks),2):
                    x = land_marks[i]
                    y = land_marks[i+1]

                    if len(mask.shape)>3:
                        mask[:, :, (y-int(self.size_mask/2)):(y+int(self.size_mask/2)), 
                                (x-int(self.size_mask/2)):(x+int(self.size_mask/2))] = False if self.keep_land_marks else True
                    else:
                        mask[:, (y-int(self.size_mask/2)):(y+int(self.size_mask/2)), 
                                (x-int(self.size_mask/2)):(x+int(self.size_mask/2))] = False if self.keep_land_marks else True
                image[mask] = 1.0

            if self.img_transform is not None:
                image = self.img_transform(image)
            
            
            return image, labels, index

    # This function provides the target label and the sensitive attribute
    def get_label(self, index):
        label_y = 0
        for i, y in enumerate(self.y_index):
            label_y += (2 ** i) * (int(self.labels[index][y]))
        
        label_z = 0
        label_landmark = []
        if (self.domain_attrs is not None) and (self.land_marks is not None):
            for i, z in enumerate(self.z_index):
                label_z += (2 ** i) * (int(self.labels[index][z]))
            for i, l in enumerate(self.landmarks_index):
                label_landmark.append(self.landmarks[index][l])
            return label_y, label_z, label_landmark
        elif self.domain_attrs is not None:
            for i, z in enumerate(self.z_index):
                label_z += (2 ** i) * (int(self.labels[index][z]))
            return label_y, label_z
        elif self.land_marks is not None:
            for i, l in enumerate(self.landmarks_index):
                label_landmark.append(self.landmarks[index][l])
            return label_y, label_landmark
        return label_y


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--num-workers", "--n", type=int, default=2)
    parser.add_argument("--data-dir", "--d", type=str, default='../data/CelebA/')
    args = parser.parse_args()
    data_dir = args.data_dir
    num_workers = args.num_workers
    
    print("================= Test Fast CelebA Dataset =================")
    data = CelebAFast(os.path.join(data_dir, 'celeba.hdf5'), ['Blond_Hair', 'Smiling'], domain_attrs=['Male'], type="test")
    loader = DataLoader(data, batch_size=256, shuffle=False, num_workers=num_workers)
    for (img, (label, domain)) in tqdm(loader):
        print(torch.unique(label))