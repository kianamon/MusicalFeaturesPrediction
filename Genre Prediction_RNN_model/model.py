import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import optim
from torch.autograd import Variable


class SequenceRegressor(nn.Module):

    def __init__(self, batch_size, hidden_size=128, num_layers=2):

        super(SequenceRegressor, self).__init__()

        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.BiLSTM = nn.LSTM(input_size=1,
                              hidden_size=self.hidden_size,
                              num_layers=self.num_layers,
                              dropout=0.5,
                              bidirectional=True)

        self.fc1 = nn.Linear(8, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size + self.hidden_size * 2, self.hidden_size)
        self.fc3 = nn.Linear(self.hidden_size, 1)
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)

    def forward(self, inputs):
        temporal_features, audio_feature = inputs
        temporal_features = temporal_features.permute(1, 0, 2).float()
        audio_features = audio_feature.float()

        lstm_output, (final_h0, final_c0) = self.BiLSTM(temporal_features, None)

        lstm_final_output = lstm_output[-1]

        x = F.relu(self.fc1(audio_features))
        x = self.dropout1(x)

        concat_featurs = torch.cat([lstm_final_output, x.squeeze()], dim=1)

        x = F.relu(self.fc2(concat_featurs))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x


class SequenceClassification(nn.Module):

    def __init__(self, batch_size, hidden_size=64, num_layers=2):

        super(SequenceClassification, self).__init__()

        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.BiLSTM = nn.LSTM(input_size=1,
                              hidden_size=self.hidden_size,
                              num_layers=self.num_layers,
                              dropout=0.5,
                              bidirectional=True)

        self.fc1 = nn.Linear(8, self.hidden_size)
        self.dropout1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(self.hidden_size + self.hidden_size * 2, self.hidden_size)
        self.fc3 = nn.Linear(self.hidden_size, 16)

        self.dropout2 = nn.Dropout(0.5)

    def forward(self, inputs):
        temporal_features, audio_feature = inputs
        temporal_features = temporal_features.permute(1, 0, 2).float()
        audio_features = audio_feature.float()

        h0 = Variable(torch.zeros(self.num_layers * 2,
                                  self.batch_size, self.hidden_size)).float()
        c0 = Variable(torch.zeros(self.num_layers * 2,
                                  self.batch_size, self.hidden_size)).float()


        lstm_output, (final_h0, final_c0) = self.BiLSTM(
            temporal_features, None)

        lstm_final_output = lstm_output[-1]

        x = F.relu(self.fc1(audio_features))
        x = self.dropout1(x)

        concat_featurs = torch.cat([lstm_final_output, x.reshape(-1, self.hidden_size)], dim=1)

        x = F.relu(self.fc2(concat_featurs))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x
