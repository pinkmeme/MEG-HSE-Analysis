#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 12:21:15 2019

@author: merzonl1 and pinkmeme
"""

import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeClassifier, LogisticRegression

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.model_selection import train_test_split

import mne
from mne.datasets import sample
from mne.decoding import (SlidingEstimator, GeneralizingEstimator, Scaler,
                          cross_val_multiscore, LinearModel, get_coef,
                          Vectorizer, CSP)
import os
import glob
import matplotlib.pyplot as plt
from mne.time_frequency import tfr_morlet

mne.set_log_level(verbose='CRITICAL')
path_preprocessed = 'derivatives/megepochs'
cond = 'experiment'
epo_cond = '-task_epo'
files = glob.glob(os.path.join(path_preprocessed, 'task', cond + '*' + epo_cond + '.fif'))

print("Start")

files_list = []
X1 = []
y1 = []
for file in files[:10]:
    epochs = mne.read_epochs(file)
    epochs.pick_types(meg=True)
    epochs.crop(tmax=3)
    X1.append(epochs.get_data())
    y1.append(epochs.events[:, 2])


X1 = tuple(X1)
y1 = tuple(y1)
X = np.swapaxes(np.concatenate(X1, axis=0), 1, 2)
y = np.concatenate(y1, axis=0) - 1

print("Scaling")

scaler = StandardScaler()
X = scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1)

X_train = Variable(torch.Tensor(X_train))
X_valid = Variable(torch.Tensor(X_valid))
y_train = Variable(torch.Tensor(y_train))
y_valid = Variable(torch.Tensor(y_valid))


class LSTM1(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, seq_length):
        super(LSTM1, self).__init__()
        self.num_layers = num_layers  # number of layers
        self.input_size = input_size  # input size
        self.hidden_size = hidden_size  # hidden state
        self.seq_length = seq_length  # sequence length

        self.dropout = nn.Dropout(p = 0.2)
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)  # lstm
        self.fc_1 = nn.Linear(hidden_size, 1)  # fully connected 1
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))  # hidden state
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))  # internal state
        # Propagate input through LSTM
        x = self.dropout(x)
        output, (hn, cn) = self.lstm(x, (h_0, c_0))  # lstm with input, hidden, and internal state
        hn = hn.view(-1, self.hidden_size)  # reshaping the data for Dense layer next
        out = self.relu(hn)
        out = self.sigmoid(self.fc_1(out))
        return out.flatten()


def get_accuracy(y_true, y_prob):
    # accuracy = torch.sum(y_true == (y_prob > 0.5)) / y_true.shape[0]
    accuracy = roc_auc_score(y_true, (y_prob > 0.5))
    return accuracy


num_epochs = 100
learning_rate = 0.01

input_size = 306
hidden_size = 100
num_layers = 1

lstm1 = LSTM1(input_size, hidden_size, num_layers, X_train.shape[1])

criterion = torch.nn.BCELoss()
optimizer = torch.optim.AdamW(lstm1.parameters(), lr=learning_rate)


print("Training")

for epoch in range(num_epochs):
    outputs = lstm1.forward(X_train)

    optimizer.zero_grad()

    loss = criterion(outputs, y_train)
    loss.backward()

    optimizer.step()

    print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))

    if epoch % 10 == 0:
        preds = lstm1.forward(X_valid)
        loss = criterion(preds, y_valid)
        acc = get_accuracy(y_valid, preds)
        print("Valid loss: %1.5f" % (loss.item()))
        print("Valid accuracy: %1.5f" % acc)


preds = lstm1.forward(X_valid)
loss = criterion(preds, y_valid)
loss.backward()
print("Valid: loss: %1.5f" % (loss.item()))