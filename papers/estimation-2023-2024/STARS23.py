from typing import Optional
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
import torch
from torch.utils.data import DataLoader, Dataset
from os import listdir
from os.path import isfile, join
import numpy as np
import librosa as lb
import matplotlib.pyplot as plt
from tqdm import tqdm
from pytorch_lightning import LightningDataModule

class STARS23(Dataset):

    def __init__(self, path_audios, fs = 16000):
        # save attributes
        self.path_audios = path_audios
        self.fs = fs
        # extract list of files
        self.list_all_files = [f for f in listdir(self.path_audios) if isfile(join(self.path_audios, f))]
                

    def __getitem__(self, index):
        #format the file path and load the file
        path = join(self.path_audios, self.list_all_files[index])
        sound, _ = lb.load(path, sr = self.fs, mono = True, res_type = "kaiser_fast")
        # replicate the clip for 10 s 
        sound = np.tile(sound, 5) 

        distance = self.list_all_files[index].split('-')[1]
        distance = float(distance[1:-1])

        
        return {
                "audio": torch.tensor(sound).float(), 
                "label": torch.tensor(distance).float(),
                "id": self.list_all_files[index]
            }
    
    def __len__(self):
        return len(self.list_all_files)
    
    def get_mean_distances(self):
        distance = 0
        for i in range(len(self.list_all_files)):
            returned_values = self.__getitem__(i)
            distance += returned_values['label']
        return distance / len(self.list_all_files)
    
    def get_distribution(self):
        distances = []
        for i in range(len(self.list_all_files)):
            returned_values = self.__getitem__(i)
            distances.append(returned_values['label'].numpy())
        distances = np.array(distances)
        plt.figure()
        plt.hist(distances, edgecolor = 'k', alpha = 0.65)
        plt.axvline(distances.mean(), color='r', linestyle='dashed', linewidth=1)
        _, max_ylim = plt.ylim()
        plt.text(distances.mean()*1.05, max_ylim*0.9, 'Mean: {:.2f} m'.format(distances.mean()))
        plt.grid(alpha = 0.2)
        plt.title("STARS23 distance distribution")
        plt.xlabel("Distance [m]")
        plt.ylabel("Occurrences")
        plt.savefig("STARS23.pdf", transparent = True)
        plt.show()
    
class STARS23DataModule(LightningDataModule):

    def __init__(self, path_dataset, batch_size, fs = 16000):
        super().__init__()
        self.path_dataset = path_dataset
        self.fs = fs
        self.batch_size = batch_size

    def prepare_data(self):
        '''
            Nothing to do
        '''
        pass
    
    def setup(self, stage = None):
        '''
            Nothing to do
        '''
        pass
    
    def train_dataloader(self):
        return DataLoader(STARS23(join(self.path_dataset, "train"), self.fs), batch_size = self.batch_size, shuffle = True, drop_last = False) 
    
    def val_dataloader(self):
        return DataLoader(STARS23(join(self.path_dataset, "val"), self.fs), batch_size = self.batch_size, shuffle = False, drop_last = False)
    
    def test_dataloader(self):
        return DataLoader(STARS23(join(self.path_dataset, "test"), self.fs), batch_size = self.batch_size, shuffle = False, drop_last = False)

    

if __name__ == "__main__":
    path_audios = 'real datasets/STARS23'
    datamodule = STARS23DataModule(path_dataset = path_audios, batch_size = 16)
    dataloader_datamodule_training = datamodule.train_dataloader()
    dataloader_datamodule_val = datamodule.val_dataloader()
    dataloader_datamodule_test = datamodule.test_dataloader()
    print("Train dataloader size: " + str(len(dataloader_datamodule_training)))
    print("Val dataloader size: " + str(len(dataloader_datamodule_val)))
    print("Test dataloader size: " + str(len(dataloader_datamodule_test)))

