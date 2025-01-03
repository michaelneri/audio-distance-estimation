import torch
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning import LightningDataModule
from os import listdir
from os.path import isfile, join
from scipy.io import loadmat
import pandas as pd
import numpy as np
import librosa as lb
from scipy.io.wavfile import write

class SyntheticDataset(Dataset):

    def __init__(self, file_path, distances_file, folds, permutation = None, fs = 24000, pathNoises = None, dBNoise = None, dist_min = None, dist_max = None):
        #initialize lists to hold file names, labels, and folder numbers
        self.file_path = file_path
        self.folds = folds
        self.distances_file = distances_file
        self.fs = fs 
        self.pathNoises = pathNoises
        self.dBNoise = dBNoise
        self.dist_min = dist_min
        self.dist_max = dist_max
        self.file_names = []
        self.labels = []
        # Read distances
        self.distances = loadmat(distances_file)['dist_s2r'].squeeze()
        if permutation is not None:
            self.permutation = loadmat(permutation)['transformed_indeces'] - 1 # From matlab to python
            self.permutation = self.permutation.squeeze()
        # Select corresponding audios
        list_all_files = [f for f in listdir(file_path) if isfile(join(file_path, f))]
        if permutation:
            list_all_files = [list_all_files[i] for i in self.permutation.tolist()]
            self.distances = self.distances[self.permutation]
        for i in range(0, len(list_all_files)):
            if i // int(len(list_all_files)/5) in folds:
                self.file_names.append(list_all_files[i])
                self.labels.append(self.distances[i])
        if self.pathNoises != None:
            # self.list_all_noises = [f for f in listdir(self.pathNoises) if isfile(join(self.pathNoises, f))]  uncomment for windows
            self.list_all_files = ['mic_sigs{:04d}.wav'.format(i) for i in range(1, 2501)]  # for Linux
    
    def __getitem__(self, index):
        #format the file path and load the file
        path = join(self.file_path, self.file_names[index])
        # sound = loadmat(path)['mic_sigs'].squeeze()
        # sound = lb.resample(sound, orig_sr = 24000, target_sr = self.fs, res_type = "polyphase")
        sound, _ = lb.load(path, sr = self.fs, mono = True, res_type = "kaiser_fast")

        # HERE SAMPLE RANDOMLY A NOISE FROM THE WHAM! DATASET AND ADD IT
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

        if self.dist_max and self.dist_min:
            distance = (self.labels[index] - self.dist_min)/(self.dist_max - self.dist_min)
        else:
            distance = self.labels[index]

        return {
                "audio": torch.tensor(sound).float(), 
                "label": torch.tensor(distance).float(),
                "id": self.file_names[index]
            }
    
    def __len__(self):
        return len(self.file_names)

class SyntheticDatasetModule(LightningDataModule):
    
    def __init__(self, file_path, distances, foldTraining, foldValidation, foldTest, batch_size, permutation = None, fs = 24000, 
                transform = None, pathNoisesTraining = None, pathNoisesVal= None, pathNoisesTest = None,  dBNoise = None, dist_min = None, dist_max = None):
        # Init 
        super().__init__()
        self.file_path = file_path
        self.distances = distances
        self.foldTraining = foldTraining 
        self.foldValidation = foldValidation
        self.batch_size = batch_size
        self.foldTest = foldTest
        self.transform = transform
        self.permutation = permutation
        self.fs = fs
        # regarding addind background noises
        self.pathNoiseTraining = pathNoisesTraining
        self.pathNoiseVal = pathNoisesVal
        self.pathNoiseTest = pathNoisesTest
        self.dBNoise = dBNoise
        # regarding distances
        self.dist_min = dist_min
        self.dist_max = dist_max

    def prepare_data(self) -> None:
        '''
            Nothing to do
        '''
        pass

    def setup(self, stage = None) -> None:
        '''
            Nothing to do
        '''
        pass

    def train_dataloader(self):
        return DataLoader(SyntheticDataset(self.file_path, self.distances, self.foldTraining, self.permutation, self.fs, self.pathNoiseTraining, self.dBNoise, self.dist_min, self.dist_max), batch_size = self.batch_size, drop_last = False)
    
    def val_dataloader(self):
        return DataLoader(SyntheticDataset(self.file_path, self.distances, self.foldValidation, self.permutation, self.fs, self.pathNoiseVal, self.dBNoise, self.dist_min, self.dist_max), batch_size = self.batch_size, shuffle = False, drop_last = False)

    def test_dataloader(self):
        return DataLoader(SyntheticDataset(self.file_path, self.distances, self.foldTest, self.permutation, self.fs, self.pathNoiseTest, self.dBNoise, self.dist_min, self.dist_max), batch_size = self.batch_size,  shuffle = False, drop_last = False)
    

##### TEST 
if __name__ == '__main__':

    file_path = 'dataset_wav/'
    distances_file = "distances.mat"
    train_set = SyntheticDataset(file_path, distances_file, list(range(0,3)))
    val_set = SyntheticDataset(file_path, distances_file, [3])
    test_set = SyntheticDataset(file_path, distances_file, [4])
    single_audio = train_set[5]
    print(single_audio['audio'].shape)
    print(single_audio['label'].shape)
    print("Train set size: " + str(len(train_set)))
    print("Val set size: " + str(len(val_set)))
    print("Test set size: " + str(len(test_set)))

    #### TEST DataModule ####

    datamodule_urban = SyntheticDatasetModule(file_path, distances_file, list(range(0,3)), [3], [4], batch_size = 16)
    dataloader_datamodule_training = datamodule_urban.train_dataloader()
    dataloader_datamodule_val = datamodule_urban.val_dataloader()
    dataloader_datamodule_test = datamodule_urban.test_dataloader()
    print("Train dataloader size: " + str(len(dataloader_datamodule_training)))
    print("Val dataloader size: " + str(len(dataloader_datamodule_val)))
    print("Test dataloader size: " + str(len(dataloader_datamodule_test)))
    
    all_labels = []
    for batch in dataloader_datamodule_training:
        all_labels.append(batch['label'].tolist())
    for batch in dataloader_datamodule_val:
        all_labels.append(batch['label'].tolist())
    for batch in dataloader_datamodule_test:
        all_labels.append(batch['label'].tolist())
    all_labels = np.hstack(all_labels)
    import matplotlib.pyplot as plt
    print(all_labels.shape)
    plt.rcParams["font.family"] = "Century Gothic"
    plt.rcParams['text.usetex'] = True
    plt.rcParams.update({'font.size': 28})
    plt.rcParams['figure.dpi'] = 600

    plt.figure(figsize = (22,12))
    plt.hist(all_labels, edgecolor = 'k', alpha = 0.7)
    plt.axvline(all_labels.mean(), color='r', linestyle='dashed', linewidth=1)
    min_ylim, max_ylim = plt.ylim()
    plt.text(all_labels.mean()*1.05, max_ylim*0.9, 'Mean: {:.2f} m'.format(all_labels.mean()))
    plt.grid(alpha = 0.2)
    plt.title("Syntethic Dataset distance distribution")
    plt.xlabel("Distance [m]")
    plt.ylabel("Occurrences")
    plt.savefig("SyntethicDatasetdistr.png", transparent = True, bbox_inches='tight')
    plt.show()





    ########################################### WITH NOISE
    file_path = 'dataset/'
    distances_file = "distances.mat"
    pathNoisesTraining = "noise_dataset/training_noise"
    pathNoisesVal= "noise_dataset/val_noise"
    pathNoisesTest = "noise_dataset/test_noise"
    permutation = "transformed_indeces.mat"

    train_set = SyntheticDataset(file_path, distances_file, list(range(0,3)), permutation, 16000, pathNoisesTraining, dBNoise = 40)
    val_set = SyntheticDataset(file_path, distances_file, [3], permutation, 16000, pathNoisesVal, dBNoise = 0)
    test_set = SyntheticDataset(file_path, distances_file, [4], permutation, 16000, pathNoisesTest, dBNoise = 10)
    single_audio = train_set[5]
    single_audio = val_set[5]
    single_audio = test_set[5]
    print(single_audio['audio'].shape)
    print(single_audio['label'].shape)
    print("Train set size: " + str(len(train_set)))
    print("Val set size: " + str(len(val_set)))
    print("Test set size: " + str(len(test_set)))

    #### TEST DataModule ####

    datamodule_urban = SyntheticDatasetModule(file_path, distances_file, list(range(0,3)), [3], [4], 16, permutation, 16000, None, pathNoisesTraining, pathNoisesVal, pathNoisesTest, 10)
    dataloader_datamodule_training = datamodule_urban.train_dataloader()
    dataloader_datamodule_val = datamodule_urban.val_dataloader()
    dataloader_datamodule_test = datamodule_urban.test_dataloader()
    print("Train dataloader size: " + str(len(dataloader_datamodule_training)))
    print("Val dataloader size: " + str(len(dataloader_datamodule_val)))
    print("Test dataloader size: " + str(len(dataloader_datamodule_test)))
