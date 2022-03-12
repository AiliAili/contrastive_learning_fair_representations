import numpy as np
import pandas as pd
from pathlib import Path

from typing import Dict

import torch
import torch.utils.data as data
from collections import Counter
import random 

class VerbDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 filename, 
                 shuffle,
                 balanced
                 ):
                
        
        # check split
        #self.dataset_type = {"train", "val", "test"}
        #assert split in self.dataset_type, "split should be one of train, dev, and test."
        #self.split = split
        #self.filename = "verb_{}_df.pickle".format(split)
        self.filename = filename
        
        # Init 
        self.X = []
        self.y = []
        self.gender_label = []
        self.image_name = []
        #self.location_label = []

        # Load preprocessed data
        print("Loading data")
        self.load_data()

        if balanced == True:
            man_idx = []
            woman_idx = []
            for i in range(0, len(self.gender_label)):
                if self.gender_label[i] == 0:
                    man_idx.append(i)
                else:
                    woman_idx.append(i)

            selected = min(len(man_idx), len(woman_idx))
            shuffle(man_idx)
            shuffle(woman_idx)
            selected_index = man_idx[:selected]+woman_idx[:selected]
            #print(selected_index[:10])

            X = [self.X[index] for index in selected_index]
            self.X = X
            y = [self.y[index] for index in selected_index]
            self.y = y
            gender_label = [self.gender_label[index] for index in selected_index]
            self.gender_label = gender_label
            image_name = [self.image_name[index] for index in selected_index]
            self.image_name = image_name


        self.X = np.array(self.X)

        self.y = np.array(self.y)
        self.gender_label = np.array(self.gender_label)
        #self.location_label = np.array(self.location_label)
        
        print("Done, loaded data shapes: {}, {}".format(self.X.shape, self.y.shape))

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.y)

    def __getitem__(self, index):
        'Generates one sample of data'
        return self.X[index], self.y[index], self.gender_label[index], self.image_name[index] #self.location_label[index]
    
    def load_data(self):
        #data = pd.read_pickle(self.data_dir+self.filename)
        data = pd.read_pickle(self.filename)
        
        self.X = list(data['features'])
        self.y = data["verb"].astype(np.int) #Verb
        self.gender_label = data["gender"].astype(np.int) # Gender
        self.image_name = list(data['image_name'])
        print(Counter(self.y))
        print(Counter(self.gender_label))
        #print("hello world", len(set(self.y))

