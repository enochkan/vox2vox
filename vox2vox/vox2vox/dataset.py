from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import glob

class CTDataset(Dataset):
    def __init__(self, datapath, transforms_):
        self.datapath = datapath
        self.transforms = transforms_
        self.samples = ['../..'+x.split('.')[4] for x in glob.glob(self.datapath + '/*.im')]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # print(self.samples[idx])
        image = h5py.File(self.samples[idx] + '.im', 'r').get('data')[()]
        mask = h5py.File(self.samples[idx] + '.seg', 'r').get('data')[()]
        # print(self.samples[idx])
        # print(image.shape)
        # print(mask.shape)
        if self.transforms:
            image, mask = self.transforms(image), self.transforms(mask)
        
        return {"A": image, "B": mask}
