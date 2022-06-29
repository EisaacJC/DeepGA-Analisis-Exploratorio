# -*- coding: utf-8 -*-
#Dataset Class

from torch.utils.data import Dataset, DataLoader
import torch
#import cv2
from PIL import Image
import numpy as np
from torchvision import transforms

#Arrays of labels
#c_labels = np.ones(918, dtype = np.int8) #Codiv
#n_labels = np.zeros(918, dtype = np.int8) #Normal
#p_labels = np.full((918,), 2) #Pneumonia
cero_labels=np.full((6903),0)
uno_labels = np.full((7877,), 1)
dos_labels = np.full((6990,), 2)
tres_labels = np.full((7141,), 3)
cuatro_labels = np.full((6824,), 4)
cinco_labels = np.full((6313,), 5)
seis_labels = np.full((6876,), 6)
siete_labels = np.full((7293,), 7)
ocho_labels = np.full((6825,), 8)
nueve_labels = np.full((6958,), 9)
#Images path
cero_root = '/content/merged/cero/cero'
uno_root = '/content/merged/uno/uno'
dos_root = '/content/merged/dos/dos'
tres_root = '/content/merged/tres/tres'
cuatro_root = '/content/merged/cuatro/cuatro'
cinco_root = '/content/merged/cinco/cinco'
seis_root = '/content/merged/seis/seis'
siete_root = '/content/merged/siete/siete'
ocho_root = '/content/merged/ocho/ocho'
nueve_root = '/content/merged/nueve/nueve'


class CovidDataset(Dataset):
  def __init__(self, root, labels, transform = None):
    self.root = root #The folder path
    self.labels = labels #Labels array
    self.transform = transform #Transform composition
  
  def __len__(self):
    return len(self.labels)
  
  def __getitem__(self, idx):
    if torch.is_tensor(idx):
      idx = idx.tolist()
    p_root = self.root[:] 
    img_name_p = p_root +" (" + str(idx+1) + ').jpg'
    #image_p = cv2.imread(img_name_p, 0)
    image_p = np.array(Image.open(img_name_p))
    [H, W] = image_p.shape
    image_p = image_p.reshape((H,W,1))
    p_label = self.labels[idx]
    sample = {'image': image_p, 'label': p_label}

    if self.transform:
      sample = self.transform(sample)

    return sample

#Class to transform image to tensor
class ToTensor(object):
  def __call__(self, sample):
    image, label = sample['image'], sample['label']

    #Swap dimmensions because:
    #       numpy image: H x W x C
    #       torch image: C x H x W
    #print(image.shape)
    image = image.transpose((2,0,1))
    #print(image.shape)
    return {'image':torch.from_numpy(image),
            'label':label}

def loading_data():
    #Loading Datasets
    cero_ds = CovidDataset(root = cero_root, labels = cero_labels, transform = transforms.Compose([ToTensor()]))
    uno_ds = CovidDataset(root = uno_root, labels = uno_labels, transform = transforms.Compose([ToTensor()]))
    dos_ds = CovidDataset(root = dos_root, labels = dos_labels, transform = transforms.Compose([ToTensor()]))
    tres_ds = CovidDataset(root = tres_root, labels = tres_labels, transform = transforms.Compose([ToTensor()]))
    cuatro_ds = CovidDataset(root = cuatro_root, labels = cuatro_labels, transform = transforms.Compose([ToTensor()]))
    cinco_ds = CovidDataset(root = cinco_root, labels = cinco_labels, transform = transforms.Compose([ToTensor()]))
    seis_ds = CovidDataset(root = seis_root, labels = seis_labels, transform = transforms.Compose([ToTensor()]))
    siete_ds = CovidDataset(root = siete_root, labels = siete_labels, transform = transforms.Compose([ToTensor()]))
    ocho_ds = CovidDataset(root = ocho_root, labels = ocho_labels, transform = transforms.Compose([ToTensor()]))
    nueve_ds = CovidDataset(root = nueve_root, labels = nueve_labels, transform = transforms.Compose([ToTensor()]))
    #Merging Covid, normal, and pneumonia Datasets
    dataset = torch.utils.data.ConcatDataset([cero_ds,uno_ds,dos_ds,tres_ds,cuatro_ds,cinco_ds,seis_ds,siete_ds,ocho_ds,nueve_ds])
    lengths = [int(len(dataset)*0.7), int(len(dataset)*0.3)]
    train_ds, test_ds = torch.utils.data.random_split(dataset = dataset, lengths = lengths)
    #print(len(dataset)
    #i = 1836
    #Testing
    #print("Length of Training Dataset: {}".format(len(train_ds)))
    #print("Length of Test Dataset: {}".format(len(test_ds)))
    #print("Shape of images as tensors: {}".format(dataset[i]['image'].shape))
    #print("Label of image i: {}".format(dataset[i]['label']))
    
    #Creating Dataloaders
    train_dl = DataLoader(train_ds, batch_size = 256, shuffle = True)
    test_dl = DataLoader(test_ds, batch_size = 256, shuffle = True)
    
    return  train_dl, test_dl

#train_dl, test_dl = loading_data()