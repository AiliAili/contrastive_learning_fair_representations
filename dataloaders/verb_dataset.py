import numpy as np
import pandas as pd
from pathlib import Path

from typing import Dict

import torch
import torch.utils.data as data
from collections import Counter
import random 
import numpy as np
from collections import defaultdict, Counter

class VerbDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 data_dir, 
                 split,
                 shuffle,
                 balanced,
                 balance_type, 
                 weight_scheme=None,
                 vanilla=False,
                 ):
                
        self.data_dir = data_dir
        
        # check split
        self.dataset_type = {"train", "val", "test"}
        #assert split in self.dataset_type, "split should be one of train, dev, and test."
        self.split = split
        self.filename = "verb_{}_df.pickle".format(split)
        
        # Init 
        self.X = []
        self.y = []
        self.gender_label = []
        self.image_name = []
        #self.location_label = []
        self.weight = []

        # Load preprocessed data
        print("Loading data")
        self.load_data()

        if balanced == True:
            assert balance_type in ["y", "g", "ratio", "class", "stratified"], "not implemented yet"
            """
                ratio:  according to its y,g combination, P(y,g)
                y:      according to its main task label y only, P(y)
                g:      according to its protected label g only, P(g)
                class:  according to its protected label within its main task label, P(g|y)
                stratified: keep the y distribution and balance g within y
            """
            
            # init a dict for storing the index of each group.

            group_idx = {}
            if balance_type == "ratio":
                group_labels = [(i,j) for i,j in zip(self.y, self.gender_label)]
            elif balance_type == "y":
                group_labels = self.y
            elif balance_type == "g":
                group_labels = self.gender_label
            elif balance_type == "class":
                group_labels = [(i,j) for i,j in zip(self.y, self.gender_label)]
            elif balance_type == "stratified":
                group_labels = [(i,j) for i,j in zip(self.y, self.gender_label)]
            else:
                pass

            for idx, group_label in enumerate(group_labels):
                group_idx[group_label] = group_idx.get(group_label, []) + [idx]


            selected_index = []
            if balance_type in ["ratio", "y", "g"]:
                # selected = min(len(man_idx), len(woman_idx))
                selected = min([len(i) for i in group_idx.values()])

                for index in group_idx.values():
                    _index = index
                    shuffle(_index)
                    selected_index = selected_index + _index[:selected]
            elif balance_type == "class":
                # balance protected groups with respect to each main task class

                # iterate each main task class
                for y in set(self.y):
                    # balance the protected group distribution
                    y_group_idx = [group_idx[(y, g)] for g in set(self.gender_label)]
                    y_selected = min([len(i) for i in y_group_idx])
                    for index in y_group_idx:
                        _index = index
                        shuffle(_index)
                        selected_index = selected_index + _index[:y_selected]
            elif balance_type == "stratified":
                # empirical distribution of y
                weighting_counter = Counter(self.y)

                # a list of (weights, actual length)
                condidate_selected = min([len(group_idx[k])/weighting_counter[k[0]] for k in group_idx.keys()])

                distinct_y_label = set(self.y)
                distinct_g_label = set(self.gender_label)
                
                # iterate each main task class
                for y in distinct_y_label:
                    selected = int(condidate_selected * weighting_counter[y])
                    for g in distinct_g_label:
                        _index = group_idx[(y,g)]
                        shuffle(_index)
                        selected_index = selected_index + _index[:selected]


            X = [self.X[index] for index in selected_index]
            self.X = X
            y = [self.y[index] for index in selected_index]
            self.y = y
            gender_label = [self.gender_label[index] for index in selected_index]
            self.gender_label = gender_label
            image_name = [self.image_name[index] for index in selected_index]
            self.image_name = image_name

        elif balanced == False:
            group_idx = {}
            inverse_weights = {}
            if weight_scheme == 'joint':
                group_labels = [(i,j) for i,j in zip(self.y, self.gender_label)]
                for idx, group_label in enumerate(group_labels):
                    group_idx[group_label] = group_idx.get(group_label, []) + [idx]

                for tem in group_idx:
                    inverse_weights[tem] = len(self.y)/float(len(group_idx[tem]))
                
                for i in range(0, len(self.y)):
                    self.weight.append(inverse_weights[(self.y[i], self.gender_label[i])])
            
            else:
                for i in range(0, len(self.y)):
                    self.weight.append(1.0)
  
        if vanilla == True:
            print('dataset leakage vanilla')
            y_one_hot = np.zeros((len(self.y), 12))
            for index, label in enumerate(self.y):
                y_one_hot[index][label] = 1
            self.y = y_one_hot

        self.X = np.array(self.X)
        self.y = np.array(self.y)
        self.gender_label = np.array(self.gender_label)
        if len(self.weight) == 0:
            self.weight = [1]*len(self.y)
        #self.location_label = np.array(self.location_label)
        
        group_labels = [(i,j) for i,j in zip(self.y, self.gender_label)]
        #print(Counter(group_labels))        
        print("Done, loaded data shapes: {}, {}".format(self.X.shape, self.y.shape))
        print(Counter(self.gender_label))

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.y)

    def __getitem__(self, index):
        'Generates one sample of data'
        return self.X[index], self.y[index], self.gender_label[index], self.image_name[index], self.weight[index] #self.location_label[index]
    
    def load_data(self):
        data = pd.read_pickle(self.data_dir+self.filename)
        
        self.X = list(data['features'])
        self.y = data["verb"].astype(np.int) #Verb
        self.gender_label = data["gender"].astype(np.int) # Gender
        self.image_name = list(data['image_name'])
        #print(Counter(self.y))
        #print(Counter(self.gender_label))
        #print("hello world", len(set(self.y))
        freq_dict = defaultdict(int)
        for i in range(0, len(self.y)):
            pro = self.y[i]
            gender = self.gender_label[i]
            freq_dict[(pro, gender)]+=1

        '''counter = 0
        for i in range(0, 211):
            print((i, 0), 100*freq_dict[(i,0)]/float(len(self.y)))
            print((i, 1), 100*freq_dict[(i,1)]/float(len(self.y)))
            if freq_dict[(i,0)] > freq_dict[(i,1)]:
                counter+=1

        print('hello world', counter)'''
        
class VerbPotentialDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 filename,
                 shuffle,
                 balanced, 
                 ):
                
        self.filename = filename
        
        # Init 
        self.X = []
        self.y = []
        self.gender_label = []
        self.image_name = []

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
        print(Counter(self.gender_label))

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.y)

    def __getitem__(self, index):
        'Generates one sample of data'
        return self.y[index], self.X[index], self.gender_label[index], self.image_name[index] #self.location_label[index]
    
    def load_data(self):
        data = pd.read_pickle(self.filename)
        
        self.X = list(data['prob'])
        self.y = data["verb"].astype(np.int) #Verb
        self.gender_label = data["gender"].astype(np.int) # Gender
        self.image_name = list(data['image_name'])


