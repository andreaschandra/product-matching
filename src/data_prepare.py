from sklearn.model_selection import train_test_split
import numpy as np
import os
from PIL import Image

PATH_IMAGE = "../../ndsc_data/training_img/training_img/"

class ShopeeDataset():
    def __init__(self, data, test, word2idx, idx2word, stage = 'train', torch = False):
        train, val = train_test_split(data, random_state = 127)
        train.reset_index(drop=True, inplace=True)
        val.reset_index(drop=True, inplace=True)
        self.word2idx = word2idx
        self.idx2word = idx2word
        self.dataset = {
            'train': (train, train.shape[0]),
            'val': (val, val.shape[0]),
            'test': (test, test.shape[0])
        }
        self.set_split(stage)
        self.torch = False
        
    def set_split(self, split='train'):
        self.data, self.length = self.dataset[split]
    
    def encode(self, text, torch = False):
        token_ids = []
        for word in text:
            try:
                token_ids.append(self.word2idx[word])
            except:
                token_ids.append(1)
                
        if torch :
          token_ids = torch.LongTensor(token_ids)
        else:
          token_ids = np.array(token_ids)
        return token_ids
    
    def decode(self, ids):
        words = []
        for id_ in ids:
            try:
                words.append(self.idx2word[id_])
            except:
                words.append('<UNK>')
                
        return words
    
    def set_fix_length(self, ids, torch = False):
        # print(ids)
        length = ids.shape[0]
        if torch :
          zeros = torch.zeros(25, dtype=torch.long)
        else:
          zeros = np.zeros(25)
        
        if length <= 25:
            zeros[:length] = ids
        else:
            zeros = ids[:25]
            
        return zeros
    
    def read_image(self, path, torch = False):
        img_arr = Image.open(path)
        img_arr = img_arr.resize((224, 224))
        
        if torch :
          img_arr = F.to_tensor(img_arr)
        else:
          img_arr = np.array(img_arr)
        
        return img_arr
    
    def __getitem__(self, idx):
        t1 = self.data.loc[idx, 'title_1_token']
        t2 = self.data.loc[idx, 'title_2_token']
        i1 = self.data.loc[idx, 'image_1']
        i2 = self.data.loc[idx, 'image_2']
        
        label = self.data.loc[idx, 'Label']
        
        t1_encode = self.encode(t1,self.torch)
        t2_encode = self.encode(t2,self.torch)
        
        t1_encode = self.set_fix_length(t1_encode, self.torch)
        t2_encode = self.set_fix_length(t2_encode, self.torch)
        
        i1_scaled = self.read_image(os.path.join(PATH_IMAGE, i1), self.torch)
        i2_scaled = self.read_image(os.path.join(PATH_IMAGE, i2), self.torch)
        
        return t1_encode, t2_encode, i1_scaled, i2_scaled, label
    
    def __len__(self):
        return self.length