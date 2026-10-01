import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlibrosa import STFT
from pytorch_lightning import LightningModule
import pandas as pd
    
class SeldNet(nn.Module):
    def __init__(self, kernels, n_grus, features_set, att_conf):
        super(SeldNet, self).__init__()
        self.n_fft = 512
        self.hop_length = 256
        self.nb_cnn2d_filt = 128
        self.pool_size = [8, 8, 2]
        self.rnn_size = [128, 128]
        self.fnn_size = 128
        self.kernels = kernels
        self.n_grus = n_grus
        self.features_set = features_set
        self.att_conf = att_conf

        # kernels "freq" [1, 3] - "time" [3, 1] - "square" [3, 3]
        if self.kernels == "freq":
            self.kernels = (1,3)
        elif self.kernels == "time":
            self.kernels = (3,1)
        elif self.kernels == "square":
            self.kernels = (3,3)
        else:
            raise ValueError
        
        self.STFT = STFT(n_fft=self.n_fft, hop_length=self.hop_length)

        # feature set "stft", "sincos", "all"
        if self.features_set == "stft":
            self.data_in = [1, (10*16000+self.n_fft)//(self.n_fft - self.hop_length) - 1,  int(self.n_fft/2)]
        elif self.features_set == "sincos":
            self.data_in = [2, (10*16000+self.n_fft)//(self.n_fft - self.hop_length) - 1,  int(self.n_fft/2)]
        elif self.features_set == "all":
            self.data_in = [3, (10*16000+self.n_fft)//(self.n_fft - self.hop_length) - 1,  int(self.n_fft/2)]
        else:
            raise ValueError
        
        # ATTENTION MAP False, "onSpec", "onAll"
        if self.att_conf == "Nothing":
            pass
        elif self.att_conf == "onSpec":
            self.heatmap = nn.Sequential(
                nn.Conv2d(in_channels = self.data_in[0], out_channels = 16,
                        kernel_size = (3,3), padding = "same", bias = False),
                nn.BatchNorm2d(16),
                nn.ELU(),
                nn.Conv2d(in_channels = 16, out_channels = 64, 
                        kernel_size = (3,3), padding = "same", bias = False),
                nn.BatchNorm2d(64),
                nn.ELU(),
                nn.Conv2d(in_channels = 64, out_channels = 1, kernel_size = 1, padding = "same"),
                nn.Sigmoid()
            )
        elif self.att_conf == "onAll":
            self.heatmap = nn.Sequential(
                nn.Conv2d(in_channels = self.data_in[0], out_channels = 16,
                        kernel_size = (3,3), padding = "same", bias = False),
                nn.BatchNorm2d(16),
                nn.ELU(),
                nn.Conv2d(in_channels = 16, out_channels = 64, 
                        kernel_size = (3,3), padding = "same", bias = False),
                nn.BatchNorm2d(64),
                nn.ELU(),
                nn.Conv2d(in_channels = 64, out_channels = self.data_in[0], kernel_size = 1, padding = "same"),
                nn.Sigmoid()
            )  
        else:
            raise ValueError         

        # First Convolutional Layers
        self.conv1 = nn.Conv2d(in_channels=self.data_in[0], out_channels=8, kernel_size=self.kernels, padding="same", bias = False)
        self.batch_norm1 = nn.BatchNorm2d(num_features=8)
        self.pool1 = nn.MaxPool2d(kernel_size=(1, self.pool_size[0]))
        self.pool1avg = nn.AvgPool2d(kernel_size=(1, self.pool_size[0]))
        
        # Second Convolutional layer
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=32, kernel_size=self.kernels, padding="same", bias = False)
        self.batch_norm2 = nn.BatchNorm2d(num_features=32)
        self.pool2 = nn.MaxPool2d(kernel_size=(1, self.pool_size[1]))
        self.pool2avg = nn.AvgPool2d(kernel_size=(1, self.pool_size[1]))

        # Third Convolutional layer
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=self.nb_cnn2d_filt, kernel_size=self.kernels, padding="same", bias = False)
        self.batch_norm3 = nn.BatchNorm2d(num_features=self.nb_cnn2d_filt)
        self.pool3 = nn.MaxPool2d(kernel_size=(1, self.pool_size[2]))
        self.pool3avg = nn.AvgPool2d(kernel_size=(1, self.pool_size[2]))

        # GRUS 2, 1, 0
        if self.n_grus == 2:
            self.gru1 = nn.GRU(input_size=int(self.data_in[-1]* self.nb_cnn2d_filt / (self.pool_size[-3]*self.pool_size[-2]*self.pool_size[-1])), hidden_size=self.rnn_size[0], bidirectional=True, batch_first = True)
            self.gru2 = nn.GRU(input_size=self.rnn_size[0]*2, hidden_size=self.rnn_size[1], bidirectional=True, batch_first = True)
        elif self.n_grus == 1:
            self.gru1 = nn.GRU(input_size=int(self.data_in[-1]* self.nb_cnn2d_filt / (self.pool_size[-3]*self.pool_size[-2]*self.pool_size[-1])), hidden_size=self.rnn_size[1], bidirectional=True, batch_first = True)
        elif self.n_grus == 0:
            self.gru_linear1 = nn.Linear(in_features = int(self.data_in[-1]* self.nb_cnn2d_filt / (self.pool_size[-3]*self.pool_size[-2]*self.pool_size[-1])), out_features = self.rnn_size[0])
            self.gru_linear2 = nn.Linear(in_features=self.rnn_size[0], out_features=self.rnn_size[1]*2)
        else:
            raise ValueError

        self.fc1 = nn.Linear(in_features=self.rnn_size[1]*2, out_features=self.fnn_size)
        self.fc2 = nn.Linear(in_features=self.fnn_size, out_features = 1)

        self.final = nn.Linear(in_features = self.data_in[-2], out_features = 1)

    def normalize_tensor(self, x):
        mean = x.mean(dim = (2,3), keepdim = True)
        std = x.std(dim = (2,3), unbiased = False, keepdim = True)
        return torch.div((x - mean), std)

    def forward(self, x):
        # features extraction
        x_real, x_imm = self.STFT(x)
        b, c, t, f = x_real.size()
        magn = torch.sqrt(torch.pow(x_real, 2) + torch.pow(x_imm, 2))
        magn = torch.log(magn**2 + 1e-7)
        previous_magn = magn

        angles_cos = torch.cos(torch.angle(x_real + 1j*x_imm))
        angles_sin = torch.sin(torch.angle(x_real + 1j*x_imm))
        magn = magn[:,:,:,:-1]
        angles_cos = angles_cos[:,:,:,:-1]
        angles_sin = angles_sin[:,:,:,:-1]

        # set up feature set
        if self.features_set == "stft":
            x = magn
        elif self.features_set == "sincos":
            x = torch.cat((angles_cos, angles_sin), dim = 1)
        elif self.features_set == "all":
            x = torch.cat((magn, angles_cos, angles_sin), dim = 1)
        else:
            raise ValueError
        
        x = self.normalize_tensor(x)

        # computation of the heatmap
        if self.att_conf == "Nothing":
            pass
        else:
            hm = self.heatmap(x)
            if self.att_conf == "onSpec":
                magn = magn * hm
                x = torch.cat((magn, angles_cos, angles_sin), dim = 1) 
                x = self.normalize_tensor(x)
            elif self.att_conf == "onAll":
                x = x * hm


        # convolutional layers
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = F.elu(x)
        x = self.pool1(x) + self.pool1avg(x)

        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = F.elu(x)
        x = self.pool2(x) + self.pool2avg(x)

        x = self.conv3(x)
        x = self.batch_norm3(x)
        x = F.elu(x)
        x = self.pool3(x) + self.pool3avg(x)

        # recurrent layers (if any)
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        if self.n_grus == 2:
            x, _ = self.gru1(x)
            x, _ = self.gru2(x)
        elif self.n_grus == 1:
            x, _ = self.gru1(x)
        else:
            x = self.gru_linear1(x)
            x = self.gru_linear2(x)

        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        
        x = x.squeeze(2) # here [batch_size, time_bins]
        rnn = x      
        x = self.final(x).squeeze()
        if self.att_conf == "Nothing":
            return x, rnn, previous_magn.detach(), None
        else:
            return x, rnn, previous_magn.detach(), hm.detach()
    

######################## Lightning Module

class SeldTrainer(LightningModule):
    def __init__(self, lr, kernels, n_grus, features_set, att_conf):
        super().__init__()

        # Hyperparameters
        self.kernels = kernels
        self.n_grus = n_grus
        self.features_set = features_set
        self.att_conf = att_conf
        self.lr = lr
        self.evaluate = torch.nn.L1Loss()
        self.loss = torch.nn.MSELoss()
        self.model = SeldNet(self.kernels, self.n_grus, self.features_set, self.att_conf)
        self.all_test_results = []

    def forward(self, x):
        x = self.model(x)
        return x
    
    def training_step(self, batch, batch_idx):
        audio, labels, _ = batch['audio'], batch['label'], batch['id']
        distance_est, time_wise_distance, _ , _ = self(audio)
        loss = self.loss(distance_est, labels)
        loss_timewise = self.loss(torch.mean(time_wise_distance, dim = -1), labels)
        final_loss = (loss + loss_timewise)/2
        self.log("train/loss", loss, prog_bar = True, on_epoch = True, on_step = False)
        self.log("train/loss_timewise", loss_timewise, prog_bar = True, on_epoch = True, on_step = False)
        self.log("train/mae", self.evaluate(distance_est, labels), on_epoch = True, prog_bar = True, on_step = False)
        return final_loss

    def validation_step(self, batch, batch_idx):
        audio, labels, _ = batch['audio'], batch['label'], batch['id']
        distance_est, time_wise_distance, _, _= self(audio)
        loss = self.loss(distance_est, labels)
        loss_timewise = self.loss(torch.mean(time_wise_distance, dim = -1), labels)
        final_loss = (loss + loss_timewise)/2
        self.log("val/loss", loss, prog_bar = True, on_epoch = True)
        self.log("val/loss_timewise", loss_timewise, prog_bar = True, on_epoch = True, on_step = False)
        self.log("val/mae", self.evaluate(distance_est, labels), on_epoch = True, prog_bar = True)
        return final_loss
    
    def test_step(self, batch, batch_idx):
        audio, labels, ids = batch['audio'], batch['label'], batch['id']
        distance_est, time_wise_distance, _ ,_ = self(audio)
        loss = self.loss(distance_est, labels)
        loss_timewise = self.loss(torch.mean(time_wise_distance, dim = -1), labels)
        self.log("test/mae_overall", self.evaluate(distance_est, labels), on_epoch = True)
        # save everything
        for element in range(labels.shape[0]):
            data = {
                    'GT': labels[element].cpu().numpy(),
                    'Pred': distance_est[element].cpu().numpy(),
                    'L1': torch.abs(distance_est[element] - labels[element]).cpu().numpy(),
                    'rL1': (torch.abs(distance_est[element] - labels[element])/labels[element]).cpu().numpy(),
                    'ID': ids[element]
                }
            self.all_test_results.append(data)
        final_loss = (loss + loss_timewise)/2
        return final_loss
    
    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr = self.lr)
        return {
           "optimizer": opt,
           "lr_scheduler": {
               "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(opt, verbose=True, patience = 5, factor = 0.2),
               "monitor": "val/loss",
               "frequency": 1
                           },
              }