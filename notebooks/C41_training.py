#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import time
from itertools import chain

import pandas as pd
import numpy as np

from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from PIL import Image
from PIL import ImageFile

import torch
import torch.nn as nn
import torch.nn.functional as nnF
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

import torchvision.models as models
from torchvision.transforms import functional as F


# In[ ]:


ImageFile.LOAD_TRUNCATED_IMAGES = True


# In[ ]:


torch.cuda.is_available()


# In[ ]:


torch.manual_seed(123)


# In[ ]:


def data_text_prep():
    d_train = pd.read_csv("../data/text_clean/train.csv")
    d_test = pd.read_csv("../data/text_clean/test.csv")

    # tokenize
    d_train.loc[:, 'title_1_token'] = d_train.title_1_pre.apply(word_tokenize)
    d_train.loc[:, 'title_2_token'] = d_train.title_2_pre.apply(word_tokenize)

    d_test.loc[:, 'title_1_token'] = d_test.title_1_pre.apply(word_tokenize)
    d_test.loc[:, 'title_2_token'] = d_test.title_2_pre.apply(word_tokenize)
    
    title_token = list(chain(*d_train.title_1_token.tolist() + d_train.title_2_token.tolist()))
    vocab_token = list(set(title_token))

    word2idx = dict((w, k) for k, w in enumerate(vocab_token, 2))
    idx2word = dict((k, w) for k, w in enumerate(vocab_token, 2))

    word2idx['<UNK>'] = 1
    idx2word[1] = '<UNK>'
    word2idx['<PAD>'] = 0
    idx2word[0] = '<PAD>'
    
    return d_train, d_test, word2idx, idx2word


# In[ ]:


class ShopeeDataset():
    def __init__(self, data, test, word2idx, idx2word):
        data['Label'] = data.Label.map({1:0, 0:1})
        train, val = train_test_split(data, random_state=127)
        train.reset_index(drop=True, inplace=True)
        val.reset_index(drop=True, inplace=True)
        self.word2idx = word2idx
        self.idx2word = idx2word
        self.dataset = {
            'train': (train, train.shape[0]),
            'val': (val, val.shape[0]),
            'test': (test, test.shape[0])
        }
        self.set_split('train')
        
    def set_split(self, split='train'):
        self.data, self.length = self.dataset[split]
    
    def encode(self, text):
        token_ids = []
        for word in text:
            try:
                token_ids.append(self.word2idx[word])
            except:
                token_ids.append(1)
        token_ids = torch.LongTensor(token_ids)
        return token_ids
    
    def decode(self, ids):
        words = []
        for id_ in ids:
            try:
                words.append(self.idx2word[id_])
            except:
                words.append('<UNK>')
                
        return words
    
    def set_fix_length(self, ids):
        length = ids.shape[0]
        zeros = torch.zeros(25, dtype=torch.long)
        
        if length <= 25:
            zeros[:length] = ids
        else:
            zeros = ids[:25]
            
        return zeros
    
    def read_image(self, path):
        img_arr = Image.open(path)
        img_arr = img_arr.resize((224, 224))
        img_arr = img_arr.convert('RGB')
        img_arr = F.to_tensor(img_arr)
        
        return img_arr
    
    def __getitem__(self, idx):
        t1 = self.data.loc[idx, 'title_1_token']
        t2 = self.data.loc[idx, 'title_2_token']
        i1 = self.data.loc[idx, 'image_1']
        i2 = self.data.loc[idx, 'image_2']
        label = self.data.loc[idx, 'Label']
        
        t1_encode = self.encode(t1)
        t2_encode = self.encode(t2)
        
        t1_encode = self.set_fix_length(t1_encode)
        t2_encode = self.set_fix_length(t2_encode)
        
        i1_scaled = self.read_image(os.path.join("../data/raw/training_img/training_img", i1))
        i2_scaled = self.read_image(os.path.join("../data/raw/training_img/training_img", i2))
        
        return t1_encode, t2_encode, i1_scaled, i2_scaled, label
    
    def __len__(self):
        return self.length


# In[ ]:


class TextEncoder(nn.Module):
    def __init__(self, num_vocab, emb_size=512, hid_size=256, num_layers=1):
        super(TextEncoder, self).__init__()
        self.network = nn.Sequential(
            nn.Embedding(num_vocab, emb_size),
            nn.LSTM(emb_size, hid_size, num_layers=num_layers, batch_first=True)
        )
        
    def forward(self, input_):
        out, (h, c) = self.network(input_)
        out = out.unsqueeze(1)
        
        return out


# In[ ]:


class ImageEncoder(nn.Module):
    def __init__(self, out_channels=256, kernel_size=(3,3)):
        super(ImageEncoder, self).__init__()
        
        mobilenet = models.mobilenet_v2(pretrained=True)
        self.model = nn.Sequential(
            mobilenet.features,
            nn.Conv2d(in_channels=1280, out_channels=out_channels, kernel_size=kernel_size)
        )
    
    def forward(self, input_):
        batch_size = input_.shape[0]
        out = self.model(input_)
        
        n_channel = out.shape[1]
        out = torch.reshape(out, (batch_size, n_channel, -1))
        
        out = out.unsqueeze(1)
        out = out.permute(0,1,3,2)
        
        return out


# In[ ]:


class BaseNetwork(nn.Module):
    def __init__(self, in_channel, kernel_size_cnn=(3,11), kernel_size_max_pool=2):
        super(BaseNetwork, self).__init__()
        
        self.base_network = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=1, kernel_size=kernel_size_cnn),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=kernel_size_cnn),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=kernel_size_max_pool),
        )
        
    def forward(self, input_):
        batch_size = input_.shape[0]
        out = self.base_network(input_)
        out = out.reshape(batch_size, -1)
        
        return out


# In[ ]:


class WrapperModel(nn.Module):
    def __init__(self):
        super(WrapperModel, self).__init__()
        self.model_text = TextEncoder(num_vocab=len(word2idx), num_layers=3)
        self.model_image = ImageEncoder()
        self.model_base = BaseNetwork(in_channel=1, kernel_size_cnn=(3,3))
        self.fc = nn.Linear(1, 1)
        
    def forward(self, t1_encode, t2_encode, i1_scaled, i2_scaled):
        feat_t1 = self.model_text(t1_encode)
        feat_t2 = self.model_text(t2_encode)
        
        feat_i1 = self.model_image(i1_scaled)
        feat_i2 = self.model_image(i2_scaled)
        
        # concatenate
        concat_1 = torch.cat((feat_t1, feat_i1), axis=3)
        concat_2 = torch.cat((feat_t2, feat_i2), axis=3)

        vec_1 = self.model_base(concat_1)
        vec_2 = self.model_base(concat_2)
        
        ed = nnF.pairwise_distance(vec_1, vec_2)
        
#         ed = euclidean_distance(vec_1, vec_2)
        
        return ed


# In[ ]:


def euclidean_distance(vec_1, vec_2):
    ed = torch.sqrt(torch.sum(torch.pow(vec_1-vec_2, 2), dim=1))
    ed = ed.reshape(-1, 1)
    return ed


# In[ ]:


def cont_loss(label, distance, margin=0.5):
    loss_contrastive = torch.mean(((1-label) * torch.pow(distance, 2)) +
                                  (label * torch.pow(torch.clamp(2 - distance, min=0), 2)))
    
    return loss_contrastive


# In[ ]:


def compute_accuracy(y_true, y_pred):
    y_pred = (y_pred > 0.5).long().squeeze()
#     y_pred = y_pred.argmax(1)
    n_correct = torch.eq(y_true, y_pred).sum().item()
    accuracy = (n_correct/y_true.shape[0])*100
    
    return accuracy


# In[ ]:


train, test, word2idx, idx2word = data_text_prep()


# In[ ]:


dataset = ShopeeDataset(train, test, word2idx, idx2word)
model = WrapperModel()
model = model.to('cuda')


# In[ ]:


num_params = sum(p.numel() for p in model.parameters())
print(f"Trainable params: {num_params:,}")


# In[ ]:


optimizer = optim.Adam(model.parameters(), lr=5e-3)
# criterion = nn.BCELoss()


# In[ ]:


dataset.set_split("train")
data_gen = DataLoader(dataset, batch_size=1)


# In[ ]:


# result_list = []
for epoch in range(1, 51):
    
    running_loss = 0
    running_loss_v = 0
    running_acc = 0
    running_acc_v = 0
    
    start = time.time()
    
    model.train()
    dataset.set_split('train')
    data_gen = DataLoader(dataset, batch_size=48)
    for batch_index, (t1_encode, t2_encode, i1_scaled, i2_scaled, label_train) in enumerate(data_gen, 1):
        
        t1_encode = t1_encode.to('cuda')
        t2_encode = t2_encode.to('cuda')
        i1_scaled = i1_scaled.to('cuda')
        i2_scaled = i2_scaled.to('cuda')
        label_train = label_train.to('cuda')
        
        optimizer.zero_grad()
        
        distance_train = model(t1_encode, t2_encode, i1_scaled, i2_scaled)
        y_pred_train = torch.sigmoid(distance_train)
        
        loss = cont_loss(label_train, distance_train, margin=0.5)
        running_loss += (loss.item() - running_loss) / batch_index
        
        accuracy = compute_accuracy(label_train, distance_train)
        running_acc += (accuracy - running_acc) / batch_index
        
        loss.backward()
        
        optimizer.step()

    model.eval()
    dataset.set_split('val')
    data_gen = DataLoader(dataset, batch_size=48)
    for batch_index, (t1_encode, t2_encode, i1_scaled, i2_scaled, label) in enumerate(data_gen, 1):
        
        t1_encode = t1_encode.to('cuda')
        t2_encode = t2_encode.to('cuda')
        i1_scaled = i1_scaled.to('cuda')
        i2_scaled = i2_scaled.to('cuda')
        label = label.to('cuda')
        
        with torch.no_grad():
            distance = model(t1_encode, t2_encode, i1_scaled, i2_scaled)
            y_pred = torch.sigmoid(distance)
        
        loss = cont_loss(label, distance, margin=0.5)
        running_loss_v += (loss.item() - running_loss_v) / batch_index
        
        accuracy = compute_accuracy(label, distance)
        running_acc_v += (accuracy - running_acc_v) / batch_index
    
#     result_list.append({"label_train": label_train.tolist(), "y_pred_train": (y_pred_train>0.5).clone().long().squeeze().detach().tolist(), "dist_train": distance_train.clone().squeeze().detach().tolist(),
#                        "label": label.tolist(), "y_pred": (y_pred>0.5).clone().long().squeeze().detach().tolist(), "dist": distance.clone().squeeze().detach().tolist()})
    
    duration = time.time() - start
    print(f"epoch: {epoch} | time: {duration:.1f}s")
    print(f"\ttrain loss: {running_loss:.2f} | train accuracy: {running_acc:.2f}")
    print(f"\tval loss: {running_loss_v:.2f} | val accuracy: {running_acc_v:.2f}")


# In[ ]:


torch.save(model.state_dict(), "../models/model.pth")

