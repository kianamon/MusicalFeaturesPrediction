#!/usr/bin/env python3

import pandas as pd
import numpy as np
from random import shuffle
import torch
from torch import nn
from torch import optim
import copy
import time

from model import SequenceRegressor
from utils import *

BATCH_SIZE = 64


def train_model(model, criterion, optimizer, num_epochs, dataloaders):

    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)


        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()

            else:
                model.eval()


            running_loss = 0.0


            for inputs, target, ids in dataloaders[phase]:


                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    temporal_data, audio_data = inputs
                    temporal_data = temporal_data.to(device)
                    audio_data = audio_data.to(device)

                    outputs = model((temporal_data, audio_data))
                    outputs = outputs.squeeze()
                    loss = criterion(outputs, target.squeeze().float())


                    if phase == 'train':
                        loss.backward()
                        optimizer.step()


                running_loss += loss.item()


            epoch_loss = running_loss / dataset_sizes[phase]
            print('{} Loss: {:.4f}'.format(phase, epoch_loss))


            if phase == 'val' and epoch_loss > best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)

    return model


if __name__ == '__main__':

    temp_feature_df = pd.read_csv('data/Temporal_Features.csv')
    audio_df = pd.read_csv('data/Audio_Features.csv')
    audio_df = audio_df.iloc[:, 1:]
    info_df = pd.read_csv('data/trackinfo.csv').iloc[:, 1:]


    track_ids = audio_df['trackID'].values
    shuffle(track_ids)
    len_ids = len(track_ids)

    train_ids = track_ids[np.arange(len_ids) < int(0.6 * len_ids)]
    val_ids = track_ids[(np.arange(len_ids) > int(0.6 * len_ids)) & (np.arange(len_ids) < int(0.9 * len_ids))]
    test_ids = track_ids[np.arange(len_ids) > int(0.9 * len_ids)]

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    train_dataset = AudioDataset(train_ids)
    val_dataset = AudioDataset(val_ids)


    dataloaders = {'train': DataLoader(train_dataset, batch_size=BATCH_SIZE, drop_last=True),
                  'val': DataLoader(val_dataset, batch_size=BATCH_SIZE, drop_last=True)}

    dataset_sizes = {'train': len(train_dataset),
                     'val': len(val_dataset)}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    model = SequenceRegressor(batch_size=BATCH_SIZE)

    criterion = nn.MSELoss()
    optimizer = optim.SGD(lr=0.001, params=model.parameters())
    optimizer = optim.Adam(lr=0.001, params=model.parameters())


    trained_model = train_model(model, criterion, optimizer, 100, dataloaders)

    torch.save(trained_model.state_dict(), 'tained_model.pth')


