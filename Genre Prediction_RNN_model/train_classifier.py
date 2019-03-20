#!/usr/bin/env python3

import pandas as pd
import numpy as np
from random import shuffle
import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
import copy
import time

from model import SequenceClassification
from utils import *

import os

BATCH_SIZE = 128


def train_model(model, criterion, optimizer, scheduler, num_epochs, dataloaders):

    if os.path.isfile('classification_tained_model.pth'):
        checkpoint = torch.load('classification_tained_model.pth')
        model.load_state_dict(checkpoint)

    since = time.time()

    loss_df = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)


        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()

            else:
                model.eval()


            running_loss = 0.0
            running_corrects = 0


            for inputs, target, ids in dataloaders[phase]:


                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    temporal_data, audio_data = inputs
                    temporal_data = temporal_data.to(device)
                    audio_data = audio_data.to(device)

                    outputs = model((temporal_data, audio_data))
                    _, preds = torch.max(outputs, 1)
                    outputs = outputs.squeeze()
                    _, target = torch.max(target.squeeze(), 1)
                    loss = criterion(outputs, target)


                    if phase == 'train':
                        loss.backward()
                        optimizer.step()


                running_loss += loss.item() * inputs[0].size(0)
                running_corrects += torch.sum(preds == target.data)


            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            
            loss_df[phase + '_loss'].append(epoch_loss)
            loss_df[phase + '_acc'].append(epoch_acc.item())


            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))


            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))


    loss_df = pd.DataFrame(loss_df)
    ax = loss_df[['train_loss', 'val_loss']].plot(figsize=(15, 8))
    ax.set_ylabel('Loss')
    fig = ax.get_figure()
    fig.savefig('loss.pdf', dpi=300)

    loss_df = pd.DataFrame(loss_df)
    ax = loss_df[['train_acc', 'val_acc']].plot(figsize=(15, 8))
    ax.set_ylabel('Acc')
    fig = ax.get_figure()
    fig.savefig('Acc.pdf', dpi=300)

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
    train_dataset = AudioDatasetClassification(train_ids)
    val_dataset = AudioDatasetClassification(val_ids)
    test_dataset = AudioDatasetClassification(test_ids)


    dataloaders = {'train': DataLoader(train_dataset, batch_size=BATCH_SIZE, drop_last=True),
                  'val': DataLoader(val_dataset, batch_size=BATCH_SIZE, drop_last=True),
                  'test': DataLoader(test_dataset, batch_size=1, drop_last=True, shuffle=False)}

    dataset_sizes = {'train': len(train_dataset),
                     'val': len(val_dataset),
                     'test': len(test_dataset)}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    model = SequenceClassification(batch_size=BATCH_SIZE).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(lr=0.01, momentum=0.9, params=model.parameters())
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    # optimizer = optim.Adam(lr=0.001, params=model.parameters())


    trained_model = train_model(model, criterion, optimizer, exp_lr_scheduler, 20, dataloaders)

    torch.save(trained_model.state_dict(), 'classification_tained_model.pth')

    preds = predict_classification(trained_model, dataloaders['test'])
    preds.to_csv('classification_prediction.csv', index=False)


