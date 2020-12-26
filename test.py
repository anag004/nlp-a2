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
from utils import *
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_sequence
import torch.nn.functional as F
import math
import sys
from functools import partial

test_filename = sys.argv[1]
output_filename = sys.argv[2]

print("Loading data...")
save_data = pickle.load(open("cs1170326_model", "rb"))
print("Done")

# Hyperparams
TRAIN_FRACTION = 0.8
np.random.seed(1729)
torch.manual_seed(1729)

# Reading the data and splitting into train and test sets
df_test = pd.read_csv(test_filename)

train_dataset = save_data["train_dataset"]
test_dataset = Dataset(df_test, 
                       word2id=train_dataset.word2id, 
                       id2word=train_dataset.id2word, 
                       word_freq=train_dataset.word_freq, 
                       threshold_len=train_dataset.threshold_len, 
                       train=False)

np.random.seed(1729)
torch.manual_seed(1729)

lstm = BiLSTM(len(train_dataset.word2id), embedding_dim=300, hidden_size=300).cuda()
lstm.load_state_dict(save_data["lstm_state_dict"])
lstm.eval()
predicted_classes = lstm.predict(test_dataset, train=False)

with open(output_filename, "w") as f:
    for x in predicted_classes:
        f.write("{}\n".format(x))

f.close()
