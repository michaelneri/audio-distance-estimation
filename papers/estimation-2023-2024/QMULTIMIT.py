import torch
from torch.utils.data import DataLoader, Dataset
from os import listdir
from os.path import isfile, join
import numpy as np
import librosa as lb
import matplotlib.pyplot as plt
from pytorch_lightning import LightningDataModule

class QMULLIMIT(Dataset):

    def __init__(self, path_audios, pathNoises = None, dBNoise = None, fs = 16000):
        # save attributes
        self.path_audios = path_audios
        self.fs = fs
        self.pathNoises = pathNoises
        self.dBNoise = dBNoise
        # extract list of files
        self.list_all_files = [f for f in listdir(self.path_audios) if isfile(join(self.path_audios, f))]
        if self.pathNoises != None:
            self.list_all_noises = [f for f in listdir(self.pathNoises) if isfile(join(self.pathNoises, f))]
                

    def __getitem__(self, index):
        #format the file path and load the file
        path = join(self.path_audios, self.list_all_files[index])
        sound, _ = lb.load(path, sr = self.fs, mono = True, res_type = "kaiser_fast")
        if self.dBNoise is not None:
            random_index_noise = np.random.randint(low = 0, high = len(self.list_all_noises))
            selected_noise_file = self.list_all_noises[random_index_noise]
            audio_noise, _ = lb.load(join(self.pathNoises, selected_noise_file), sr = self.fs, mono = True, res_type = "kaiser_fast")

            RMS_s = np.sqrt(np.mean(np.power(sound,2)))
            if self.dBNoise == "Random": # extractly randomly an SNR
                random_SNR = np.random.rand() * 50
                RMS_n = np.sqrt(np.power(RMS_s,2) / np.power(10, random_SNR/10))
            else:
                RMS_n = np.sqrt(np.power(RMS_s,2) / np.power(10, self.dBNoise/10))

            RMS_n_current = np.sqrt(np.mean(np.power(audio_noise,2)))

            
            audio_noise = audio_noise * (RMS_n / RMS_n_current)

            sound = sound.squeeze() + audio_noise
        else:
            sound = sound.squeeze()

        distance = self.list_all_files[index].split('_')[2]
        distance = float(distance[:-1])

        
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
        plt.title("QMULTIMIT distance distribution")
        plt.xlabel("Distance [m]")
        plt.ylabel("Occurrences")
        plt.savefig("QMULTIMIT test.pdf", transparent = True)
        plt.show()

        
    
class QMULLIMITDataModule(LightningDataModule):

    def __init__(self, path_dataset, batch_size, pathNoisesTraining = None, pathNoisesVal= None, pathNoisesTest = None, db = None, fs = 16000):
        super().__init__()
        self.path_dataset = path_dataset
        self.fs = fs
        self.batch_size = batch_size
        # regarding addind background noises
        self.pathNoiseTraining = pathNoisesTraining
        self.pathNoiseVal = pathNoisesVal
        self.pathNoiseTest = pathNoisesTest
        self.dBNoise = db

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
        return DataLoader(QMULLIMIT(join(self.path_dataset, "train",), pathNoises = self.pathNoiseTraining, dBNoise = self.dBNoise, fs = self.fs), batch_size = self.batch_size, shuffle = True, drop_last = False) 
    
    def val_dataloader(self):
        return DataLoader(QMULLIMIT(join(self.path_dataset, "val"), pathNoises = self.pathNoiseVal, dBNoise = self.dBNoise, fs = self.fs), batch_size = self.batch_size, shuffle = False, drop_last = False)
    
    def test_dataloader(self):
        return DataLoader(QMULLIMIT(join(self.path_dataset, "test"), pathNoises = self.pathNoiseTest, dBNoise = self.dBNoise, fs = self.fs), batch_size = self.batch_size, shuffle = False, drop_last = False)
    
    def all_dataloader(self):
        return DataLoader(QMULLIMIT(join(self.path_dataset, "all"), pathNoises = self.pathNoiseTest, dBNoise = self.dBNoise, fs = self.fs), batch_size = self.batch_size, shuffle = False, drop_last = False)

    

if __name__ == "__main__":
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams['text.usetex'] = True
    path_audios = 'real datasets/QMUL_TIMIT'
    pathNoisesTraining = "noise_dataset/training_noise"
    pathNoisesVal = "noise_dataset/val_noise"
    pathNoisesTest = "noise_dataset/test_noise"
    datamodule = QMULLIMITDataModule(path_dataset = path_audios, batch_size = 4, pathNoisesTraining = pathNoisesTraining, pathNoisesVal = pathNoisesVal, pathNoisesTest = pathNoisesTest, db = 10)
    dataloader_datamodule_training = datamodule.train_dataloader()
    dataloader_datamodule_val = datamodule.val_dataloader()
    dataloader_datamodule_test = datamodule.test_dataloader()
    print("Train dataloader size: " + str(len(dataloader_datamodule_training)))
    print("Val dataloader size: " + str(len(dataloader_datamodule_val)))
    print("Test dataloader size: " + str(len(dataloader_datamodule_test)))
    train_ds = QMULLIMIT(join(path_audios, "train"), pathNoises = pathNoisesTraining, dBNoise = 10, fs = 16000)
    val_ds = QMULLIMIT(join(path_audios, "val"), pathNoises = pathNoisesTraining, dBNoise = 10, fs = 16000)
    test_ds = QMULLIMIT(join(path_audios, "test"), pathNoises = pathNoisesTraining, dBNoise = 10, fs = 16000)
    all_ds = QMULLIMIT(join(path_audios, "all"), pathNoises = pathNoisesTraining, dBNoise = 10, fs = 16000)
    # print distances distribution
    all_ds.get_distribution()

