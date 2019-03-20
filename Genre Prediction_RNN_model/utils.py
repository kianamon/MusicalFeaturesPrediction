from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import OneHotEncoder


def normalize_df(df, axis=0):
    track_id = df['trackID']
    df.drop(labels='trackID', axis=1)
    mean = df.mean(axis=axis)
    std = df.std(axis=axis)
    df = (df - mean) / std
    df['trackID'] = track_id
    return df


def normalize_series(s, scaler=None):
    if scaler == 'minmax':
        out = (s - s.min()) / (s.max() - s.min())
    elif scaler == 'standard':
        out = (s - s.mean()) / s.std()
    else:
        out = s
    return out


def normalize_temporal_features(df):
    track_id = df['trackID']
    df.drop(labels='trackID', axis=1)
    denom = df.max(axis=1) - df.min(axis=1)
    df = df.subtract(df.min(axis=1), axis='rows').divide(denom, axis='rows')
    df['trackID'] = track_id
    return df


class AudioDataset(Dataset):

    def __init__(self, ids):

        self.temporal_features = pd.read_csv('data/Temporal_Features.csv')
        self.temporal_features = normalize_temporal_features(
            self.temporal_features)

        self.audio_features = pd.read_csv(
            'data/Audio_Features.csv').iloc[:, 1:]
        self.audio_features = normalize_df(self.audio_features)

        self.track_ids = ids
        self.track_info = pd.read_csv('data/trackinfo.csv').iloc[:, 1:]
        self.track_info = self.track_info[self.track_info['trackID'].isin(
            self.track_ids)][['trackID', 'interest']]
        self.track_info['interest'] = normalize_series(
            self.track_info['interest'], scaler='standard')

    def __len__(self):
        return len(self.track_ids)

    def __getitem__(self, idx):
        track_ids = self.track_ids[idx]
        if isinstance(track_ids, np.int64):
            track_ids = np.array([track_ids])
        sample_temporal_features = self.temporal_features[self.temporal_features['trackID'].isin(
            track_ids)].drop(labels='trackID', axis=1)
        sample_audio_features = self.audio_features[self.audio_features['trackID'].isin(
            track_ids)].drop(labels='trackID', axis=1)
        sample_target = self.track_info[self.track_info['trackID'].isin(
            track_ids)].drop(labels='trackID', axis=1)

        sample_temporal_features = torch.tensor(
            sample_temporal_features.values).reshape(224, 1)
        sample_audio_features = torch.tensor(sample_audio_features.values)
        sample_target = torch.tensor(sample_target.values)

        return (sample_temporal_features, sample_audio_features), sample_target, track_ids


class AudioDatasetClassification(Dataset):

    def __init__(self, ids):

        self.temporal_features = pd.read_csv('data/Temporal_Features.csv')
        self.temporal_features = normalize_temporal_features(
            self.temporal_features)

        self.audio_features = pd.read_csv(
            'data/Audio_Features.csv').iloc[:, 1:]
        self.audio_features = normalize_df(self.audio_features)

        self.track_ids = ids
        self.track_info = pd.read_csv('data/trackinfo.csv').iloc[:, 1:]
        self.encoder = OneHotEncoder()
        self.encoder.fit(self.track_info['genre_top'].values.reshape(-1, 1))
        self.track_info = self.track_info[self.track_info['trackID'].isin(
            self.track_ids)][['trackID', 'genre_top']]


    def __len__(self):
        return len(self.track_ids)

    def __getitem__(self, idx):
        track_ids = self.track_ids[idx]
        if isinstance(track_ids, np.int64):
            track_ids = np.array([track_ids])
        sample_temporal_features = self.temporal_features[self.temporal_features['trackID'].isin(
            track_ids)].drop(labels='trackID', axis=1)
        sample_audio_features = self.audio_features[self.audio_features['trackID'].isin(
            track_ids)].drop(labels='trackID', axis=1)
        sample_target = self.track_info[self.track_info['trackID'].isin(
            track_ids)].drop(labels='trackID', axis=1)

        sample_temporal_features = torch.tensor(
            sample_temporal_features.values).reshape(224, 1)
        sample_audio_features = torch.tensor(sample_audio_features.values)
        sample_target = self.encoder.transform(sample_target.values).toarray()
        sample_target = torch.tensor(sample_target)

        return (sample_temporal_features, sample_audio_features), sample_target, track_ids



def predict(model, dataloader, scaler=None):

    prediction = []
    for inputs, target, ids in dataloader:

        model.eval()
        output = model(inputs).detach().numpy().squeeze()
        ids = ids.numpy().squeeze()
        
        if scaler == 'minmax':
            output = output * (3293557 - 2) + 2
        elif scaler == 'standard':
            output = output * (19017.430891209708) + 3541.3102069923248

        pred = zip(ids, output)
        prediction.extend(pred)
    
    prediction = pd.DataFrame.from_records(prediction, columns=['trackID', 'prediction'])
    return prediction


def predict_classification(model, dataloader):
    running_corrects = 0
    prediction = []
    for inputs, target, ids in dataloader:

        model.eval()
        output = model(inputs).detach()
        preds = torch.argmax(output)
        ids = ids.numpy().squeeze()
        running_corrects += torch.sum(preds == torch.argmax(target))
        prediction.append((ids.item(), preds.item()))
    
    test_accuracy = running_corrects.float() / len(dataloader)
    print("test accuracy: {:.4f}".format(test_accuracy))
    prediction = pd.DataFrame.from_records(prediction, columns=['trackID', 'predictied_genre'])
    return prediction