from typing import Optional
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
import torch
from torch.utils.data import DataLoader, Dataset
from os import listdir
from os.path import isfile, join
import numpy as np
import librosa as lb
import matplotlib.pyplot as plt

class VoiceHome2(Dataset):

    def __init__(self, path_annotations, path_audios, filenames, distances, fs = 16000):
        # save attributes
        self.path_annotations = path_annotations
        self.path_audios = path_audios
        self.fs = fs
        self.filenames = filenames
        self.distances = distances
                

    def __getitem__(self, index):
        #format the file path and load the file
        path = join(self.path_audios, self.filenames[index])
        sound, _ = lb.load(path, sr = self.fs, mono = False, res_type = "kaiser_fast")
        sound = sound[0,:]
        # cut a clip if longer than 10s
        if len(sound) > self.fs * 10:
            sound = sound[:self.fs*10]
        else: # otherwise zero pad
            temp = np.zeros(self.fs * 10)
            temp[:len(sound)] = sound
            sound = temp

        
        return {
                "audio": torch.tensor(sound).float(), 
                "label": torch.tensor(self.distances[index]).float(),
                "id": self.filenames[index]
            }
    
    def __len__(self):
        return len(self.filenames)
    
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
        plt.title("VoiceHome distance distribution")
        plt.xlabel("Distance [m]")
        plt.ylabel("Occurrences")
        plt.savefig("VoiceHome2.pdf", transparent = True)
        plt.show()
    
class VHDataModule(LightningDataModule):

    def __init__(self, npz_file, path_annotations, path_audios, batch_size):
        super().__init__()
        self.npz_file = npz_file
        self.path_annotations = path_annotations
        self.path_audios = path_audios
        self.batch_size = batch_size
        
        # read npz and split
        read_file = np.load(self.npz_file)
        # training files and distances
        self.training_files = read_file['arr_0']
        self.training_distances = read_file['arr_1']
        self.validation_files = read_file['arr_2']
        self.validation_distances = read_file['arr_3']
        self.testing_files = read_file['arr_4']
        self.testing_distances = read_file['arr_5']
        


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
        return DataLoader(VoiceHome2(self.path_annotations, self.path_audios, self.training_files, self.training_distances), batch_size = self.batch_size, shuffle = True, drop_last = False)
    
    def val_dataloader(self):
        return DataLoader(VoiceHome2(self.path_annotations, self.path_audios, self.validation_files, self.validation_distances), batch_size = self.batch_size, shuffle = False, drop_last = False)
    
    def test_dataloader(self):
        return DataLoader(VoiceHome2(self.path_annotations, self.path_audios, self.testing_files, self.testing_distances), batch_size = self.batch_size, shuffle = False, drop_last = False)
    

    

if __name__ == "__main__":
    path_annotations = 'real datasets/VoiceHome2/voiceHome-2_corpus_1.0/annotations/rooms'
    path_audios = 'real datasets/VoiceHome2/voiceHome-2_corpus_1.0/audio/noisy'
    file_npz = "VoiceHome2_splitted.npz"
    datamodule = VHDataModule(file_npz, path_annotations, path_audios, batch_size = 16)
    dataloader_datamodule_training = datamodule.train_dataloader()
    dataloader_datamodule_val = datamodule.val_dataloader()
    dataloader_datamodule_test = datamodule.test_dataloader()
    print("Train dataloader size: " + str(len(dataloader_datamodule_training)))
    print("Val dataloader size: " + str(len(dataloader_datamodule_val)))
    print("Test dataloader size: " + str(len(dataloader_datamodule_test)))


