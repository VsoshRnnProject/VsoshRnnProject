import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import gensim
import nltk
import gc

import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence

class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RNNModel, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Sequential(torch.nn.Linear(hidden_size, 1), torch.nn.Sigmoid())

    def forward(self, packed_input):
        packed_output, (hidden, _) = self.rnn(packed_input)  # RNN принимает PackedSequence
        output = self.fc(hidden[-1])  # Используем последний скрытый слой
        return output

def text_to_vec(text):
    tokens = nltk.tokenize.word_tokenize(text.lower())
    vector = [w2v.wv[token].tolist() if token in w2v.wv else w2v.wv['notfound'].tolist() for token in tokens]
    return vector



model = RNNModel(input_size=100, hidden_size=45, num_layers=1, output_size=1)

model.load_state_dict(torch.load("C:\My_folder\For_VS_Code\VsoshRnnProject-main\RNN.pth"))
model.eval()
w2v = gensim.models.Word2Vec.load("C:\My_folder\For_VS_Code\VsoshRnnProject-main\W2V.model")

# Получение пользовательского ввода
input_for_model = input("Введите строку:\n")
# input_for_model = "Hello, how are you?"
# input_for_model = "' OR 1 = 1 #"


# получение предсказания
input_vector = text_to_vec(input_for_model)
input_vector = torch.tensor(input_vector)

predict = model(input_vector)
print(predict)
ans = float(predict.item())
print(f'Ans: {ans:.8f}')