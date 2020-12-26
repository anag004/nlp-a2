# Imports
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sklearn
import scipy
from nltk.tokenize import word_tokenize
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pdb
import math
from tqdm import tqdm
from utils import *
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_sequence
import torch.nn.functional as F
import math
import sys

train_filename = sys.argv[1]

# Hyperparams
TRAIN_FRACTION = 0.8
COMPUTE_VAL = False

set_deterministic()

# Reading the data and splitting into train and test sets
df_train = pd.read_csv(train_filename)
df_train = sklearn.utils.shuffle(df_train, random_state=1729)

if COMPUTE_VAL:
    df_test = pd.read_csv("data/val.csv")

train_dataset = Dataset(df_train, threshold_freq=5, threshold_len=2500)

if COMPUTE_VAL:
    test_dataset = Dataset(df_test, 
                        word2id=train_dataset.word2id, 
                        id2word=train_dataset.id2word, 
                        word_freq=train_dataset.word_freq,
                        threshold_len=train_dataset.threshold_len)

NUM_EPOCHS = 50
LR = 0.01

print("Training the embedding bag")

embedding_bag = EmbeddingBagModel(len(train_dataset.word2id), embedding_dim=300).cuda()
train_loader = DataLoader(train_dataset, batch_size=500, collate_fn=collate_fn)
criterion = nn.CrossEntropyLoss(weight=get_class_weights(train_dataset)).cuda()
optimizer = torch.optim.Adam(embedding_bag.parameters(), lr=LR, weight_decay=1e-6)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.98)	

for epoch in range(NUM_EPOCHS):
    embedding_bag.train()
    avg_loss = 0
    num_iters = 0

    for data in train_loader:
        embedding_bag.zero_grad()

        logits = embedding_bag(data["sequences"].cuda(), data["offsets"].cuda())
        loss = criterion(logits, data["labels"].cuda())

        avg_loss += loss.item()
        num_iters += 1

        loss.backward()
        optimizer.step()
        
    scheduler.step()

    avg_loss /= num_iters

    embedding_bag.eval()

    if COMPUTE_VAL:
        predicted_classes = embedding_bag.predict(test_dataset).cpu()
        val_acc = eval_acc(predicted_classes, test_dataset.classes)

    predicted_classes = embedding_bag.predict(train_dataset).cpu()
    train_acc = eval_acc(predicted_classes, train_dataset.classes)

    if COMPUTE_VAL:
        print("Epoch : {} Avg Loss: {:.4} Val Acc: {} Train Acc: {}".format(epoch, avg_loss, val_acc, train_acc))
    else:
        print("Epoch : {} Avg Loss: {:.4} Train Acc: {}".format(epoch, avg_loss, train_acc))

print("Training the BiLSTM")

NUM_EPOCHS = 4
LR = 0.001

lstm = BiLSTM(len(train_dataset.word2id), embedding_dim=300, hidden_size=300, embedding_bag=embedding_bag).cuda()
train_loader = DataLoader(train_dataset, batch_size=128, collate_fn=collate_fn)
criterion = nn.CrossEntropyLoss(weight=get_class_weights(train_dataset)).cuda()
optimizer = torch.optim.Adam(lstm.parameters(), lr=LR)

for epoch in range(NUM_EPOCHS):
    lstm.train()
    avg_loss = 0
    num_iters = 0

    pbar = tqdm(train_loader)

    for data in pbar:
        lstm.zero_grad()

        logits = lstm(data["sequences"].cuda(), data["offsets"].cuda())
        loss = criterion(logits, data["labels"].cuda())

        avg_loss += loss.item()
        num_iters += 1

        pbar.set_description("Loss: {:.4}".format(loss.item()))

        loss.backward()
        optimizer.step()
        torch.cuda.empty_cache()

    avg_loss /= num_iters

    lstm.eval()
    if COMPUTE_VAL:
        predicted_classes = lstm.predict(test_dataset).cpu()
        val_acc = eval_acc(predicted_classes, test_dataset.classes)

    predicted_classes = lstm.predict(train_dataset).cpu()
    train_acc = eval_acc(predicted_classes, train_dataset.classes)

    if COMPUTE_VAL:
        print("Epoch : {} Avg Loss: {:.4} Val Acc: {} Train Acc: {}".format(epoch, avg_loss, val_acc, train_acc))
    else:
        print("Epoch : {} Avg Loss: {:.4} Train Acc: {}".format(epoch, avg_loss, train_acc))

save_data = {}
save_data["train_dataset"] = train_dataset
save_data["lstm_state_dict"] = lstm.state_dict()

pickle.dump(save_data, open("cs1170326_model", 'wb'))
