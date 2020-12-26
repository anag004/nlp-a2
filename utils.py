import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sklearn
import scipy
from nltk.tokenize import word_tokenize
import string
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
import pdb
import math
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.metrics import f1_score, accuracy_score, recall_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import BaggingClassifier
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression, SGDClassifier, RidgeClassifier
from sklearn import preprocessing
from tqdm import tqdm
import nltk
import re
from torch.utils.data import Dataset as TorchDataset
import pickle
import os
import torch
import sys
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_sequence
from torch.utils.data import DataLoader
from functools import partial
import random

classes_to_id = {}
classes_to_id['Meeting and appointment'] = 1
classes_to_id['For circulation'] = 2
classes_to_id['Selection committee issues'] = 3
classes_to_id['Selection committe issues'] = 3 
classes_to_id['Policy clarification/setting'] = 4
classes_to_id['Recruitment related'] = 5
classes_to_id['Assessment related'] = 6
classes_to_id['Other'] = 7

def nltk_tag_to_wordnet_tag(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:          
        return None

class Dataset(TorchDataset):
    def __init__(self, df, id2word=None, word2id=None, word_freq=None, train=True, threshold_freq=5, threshold_len=2500):
        self.df = df
        self.threshold_freq = threshold_freq
        self.threshold_len = threshold_len
        self.train = train
        self.train_vectorizer = None
        self.classes = None
        self.classes_freq = None
        
        if self.train:
            self.parse_classes()
        
        self.content_corpus = self.parse_content(df['Content'])
        self.subject_corpus = self.parse_content(df['Subject']) 
        self.combined_corpus = [a * 3 +  b for a, b in zip(self.subject_corpus, self.content_corpus)]
        
        if word2id is None:
            self.word2id, self.id2word, self.word_freq = self.construct_vocab(self.combined_corpus)
        else:
            self.word2id = word2id
            self.id2word = id2word
            self.word_freq = word_freq

        self.sequences = self.construct_sequences(self.combined_corpus)

    def construct_vocab(self, corpus):
        word_freq = {}

        for sentence in tqdm(corpus, desc="Counting words"):
            for word in sentence:
                if word not in word_freq:
                    word_freq[word] = 0

                word_freq[word] += 1

        word2id = {}
        id2word = {}
        id2word[0] = "UNK" # UNK has ID = 0
        word2id["UNK"] = 0

        for word, freq in tqdm(word_freq.items(), desc="Encoding words"):    
            if word == "UNK" or word_freq[word] < self.threshold_freq:
                continue 
            else:
                word_id = len(word2id)
                word2id[word] = word_id
                id2word[word_id] = word

        return word2id, id2word, word_freq

    def construct_sequences(self, corpus):
        result = []

        for sentence in tqdm(corpus, desc="Encoding sentences"):
            encoded_sentence = []

            for word in sentence:
                if word not in self.word2id:
                    word = "UNK"
                else:
                    encoded_sentence.append(self.word2id[word])
            
            result.append(encoded_sentence)

        return result

    def parse_content(self, df_series):
        data = []
        
        for doc in tqdm(df_series.values, desc="Parsing content"):
            if not (doc is np.nan):
                data.append(self.word_process(doc))
            else:
                data.append(["UNK"])
                
        return data
    
    def word_process(self, sentence): 
        # Adapted from https://medium.com/@gaurav5430/using-nltk-for-lemmatizing-sentences-c1bfff963258
        
        lemmatizer = WordNetLemmatizer()
        nltk_tagged = nltk.pos_tag(nltk.word_tokenize(sentence)) 
        wordnet_tagged = map(lambda x: (x[0], nltk_tag_to_wordnet_tag(x[1])), nltk_tagged)
        
        lemmatized_sentence = []
        for word, tag in wordnet_tagged:
            word = re.sub(r'', '<.*?>', word) # Remove HTML
            word = re.sub(r'=\n', '', word) # Remove word spilling across lines
            word = re.sub(r'^https?:\/\/.*[\r\n]*', '', word, flags=re.MULTILINE) # Remove URLs
            word = re.sub(r'\d+', '', word) # Remove numbers
            
            if tag is None:
                # if there is no available tag, append the token as is
                lemmatized_sentence.append(word)
            else:        
                # else use the tag to lemmatize the token
                lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))
        
        if self.threshold_len < len(lemmatized_sentence):
            lemmatized_sentence = lemmatized_sentence[:self.threshold_len]
        
        return lemmatized_sentence
        
    def parse_classes(self):
        self.classes = []
        self.classes_freq = [0 for _ in range(7)]
        
        for cname in self.df['Class'].tolist():
            self.classes.append(classes_to_id[cname])
            self.classes_freq[classes_to_id[cname] - 1]  += 1
            
        self.classes = np.array(self.classes)
        self.classes_freq /= np.sum(self.classes_freq)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        if self.train:
            return self.sequences[idx], self.classes[idx]
        else:
            return self.sequences[idx], None

def collate_fn(batch, train=True):
    batch_data = { "sequences" : [], "offsets" : [], "labels" : [] }
    starting_idx = 0

    for seq, lbl in batch:
        batch_data["sequences"] = batch_data["sequences"] + seq 
        batch_data["offsets"].append(starting_idx)
	
        if train:
            batch_data["labels"].append(lbl)

        starting_idx += len(seq)

    batch_data["sequences"] = torch.Tensor(batch_data["sequences"]).long() 
    batch_data["offsets"] = torch.Tensor(batch_data["offsets"]).long() 
    
    if train:
        batch_data["labels"] = torch.Tensor(batch_data["labels"]).long() - 1 

    return batch_data

def eval_acc(predicted, actual):
    return (
        accuracy_score(actual, predicted), 
        recall_score(actual, predicted, average="macro")
    )

def set_deterministic():
    np.random.seed(1729)
    torch.manual_seed(1729)
    random.seed(1729)
    torch.backends.cudnn.benchmark = False

class EmbeddingBagModel(nn.Module):
    def __init__(self, num_embeddings, embedding_dim=300, mode="mean", num_classes=7):
        super(EmbeddingBagModel, self).__init__()

        self.embedding_bag = nn.EmbeddingBag(num_embeddings=num_embeddings,
                                             embedding_dim=embedding_dim,
                                             mode=mode)
        self.drop = nn.Dropout()
        self.fc = nn.Linear(embedding_dim, num_classes)

    def forward(self, seq, offsets):
        return self.fc(self.drop(self.embedding_bag(seq, offsets)))

    def predict(self, test_dataset, return_logits=False, train=True):
        test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False, collate_fn=partial(collate_fn, train=train))

        for data in test_loader:
            logits = self(data["sequences"].cuda(), data["offsets"].cuda())

            if return_logits:
                return logits
            else:
                return torch.argmax(logits, dim=1).cpu() + 1

def predict_ensemble(test_dataset, models):
    logits = F.softmax(models[0].predict(test_dataset, return_logits=True), dim=1)

    for model in models[1:]:
        model.eval()
        logits += F.softmax(model.predict(test_dataset, return_logits=True), dim=1)

    return torch.argmax(logits, dim=1) + 1

def get_class_weights(dataset):
    classes_freq = dataset.classes_freq

    return torch.Tensor(np.max(classes_freq) / classes_freq)

class BiLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim=300, hidden_size=300, num_classes=7, embedding_bag=None):
        super(BiLSTM, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(num_embeddings=self.vocab_size, 
                                      embedding_dim=self.embedding_dim)
        self.bn = nn.BatchNorm1d(2 * hidden_size)

        if embedding_bag is not None:
            self.embedding.weight.data = embedding_bag.embedding_bag.weight.data
        
        self.lstm = nn.LSTM(input_size=embedding_dim, 
                            hidden_size=hidden_size, 
                            batch_first=True, 
                            bidirectional=True)
        self.fc = nn.Linear(2 * hidden_size, num_classes)
        
    def forward(self, seq, offsets):
        embedding_seq = []
        
        for i, start in enumerate(offsets):
            if i == offsets.shape[0] - 1:
                end = seq.shape[0]
            else:
                end = offsets[i+1]
                
            embeddings = self.embedding(seq[start:end])
            embedding_seq.append(embeddings)
            
        packed_input = pack_sequence(embedding_seq, enforce_sorted=False)
        packed_output, _ = self.lstm(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        output, _ = torch.max(output, dim=1)
        output = self.bn(output)

        return self.fc(output)
    
    def predict(self, test_dataset, return_logits=False, train=True):
        test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False, collate_fn=partial(collate_fn, train=train))
        
        logits = []

        for data in test_loader:
            logits.append(self(data["sequences"].cuda(), data["offsets"].cuda()))

        logits = torch.cat(logits)
            
        if return_logits:
            return logits
        else:
            return torch.argmax(logits, dim=1) + 1

