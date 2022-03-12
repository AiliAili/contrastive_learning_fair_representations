import numpy as np
import pandas as pd
from pathlib import Path

from typing import Dict

import torch
import torch.utils.data as data

from collections import defaultdict, Counter


def full_label_data(df, tasks):
    selected_rows = np.array([True]*len(df))
    for task in tasks:
        selected_rows = selected_rows & df[task].notnull().to_numpy()
    return selected_rows

class BiosDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 data_dir, 
                 split, 
                 balanced,
                 balance_type,
                 embedding_type = "bert_avg_SE",
                 shuffle=None, 
                 vanilla=False,
                 ):
                
        self.data_dir = data_dir
        
        # check split
        self.dataset_type = {"train", "dev", "test"}
        assert split in self.dataset_type, "split should be one of train, dev, and test."
        self.split = split
        self.filename = "bios_{}_df.pkl".format(split)
        
        # check embedding type
        assert embedding_type in ("avg", "cls", "bert_avg_SE"), "Embedding should either be avg or cls."
        """
            avg and cls are sentence representations from BERT encoders
            bert_avg_SE are sentecne representations from fine-tuned BERT encoders
        """
        if embedding_type in ("avg", "cls"):
            self.embedding_type = "train_{}_data".format(embedding_type)
        else:
            self.embedding_type = embedding_type

        # Init 
        self.X = []
        self.y = []
        self.gender_label = []
        #self.location_label = []

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
        if vanilla == True:
            print('dataset leakage vanilla')
            y_one_hot = np.zeros((len(self.y), 28))
            for index, label in enumerate(self.y):
                #print(index, label)
                y_one_hot[index][label] = 1
            self.y = y_one_hot

        self.X = np.array(self.X)
        if len(self.X.shape) == 3:
            self.X = np.concatenate(list(self.X), axis=0)
        self.y = np.array(self.y)
        self.gender_label = np.array(self.gender_label)
        #self.location_label = np.array(self.location_label)
        
        print("Done, loaded data shapes: {}, {}".format(self.X.shape, self.y.shape))
        print(Counter(self.y), Counter(self.gender_label))

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.y)

    def __getitem__(self, index):
        'Generates one sample of data'
        return self.X[index], self.y[index], self.gender_label[index] #self.location_label[index]
    
    def load_data(self):
        data = pd.read_pickle(self.data_dir+self.filename)

        freq_dict = defaultdict(int)
        
        self.X = list(data[self.embedding_type])
        self.y = data["profession_class"].astype(np.int) #Profession
        #self.location_label = data["econ_class"].astype(np.int) # location
        self.gender_label = data["gender_class"].astype(np.int) # Gender
        #print("hello world", len(set(self.y)))

        for i in range(0, len(self.y)):
            pro = self.y[i]
            gender = self.gender_label[i]
            freq_dict[(pro, gender)]+=1
        
        #for i in range(0, 28):
        #    print((i, 0), 100*freq_dict[(i,0)]/float(len(self.y)))
        #    print((i, 1), 100*freq_dict[(i,1)]/float(len(self.y)))

        #print(Counter(self.gender_label))
        filtered_x = []
        filtered_y = []
        filtered_gender_label = []
        for i in range(0, len(self.y)):
            if self.y[i] == 13:
                filtered_x.append(self.X[i])
                filtered_y.append(0)
                filtered_gender_label.append(self.gender_label[i])
            elif self.y[i] == 25:
                filtered_x.append(self.X[i])
                filtered_y.append(1)
                filtered_gender_label.append(self.gender_label[i])

        self.X = filtered_x
        self.y = filtered_y
        self.gender_label = filtered_gender_label
        return 0


class BiosPotentialDataset(torch.utils.data.Dataset):
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
        return self.y[index], self.X[index], self.gender_label[index]
    
    def load_data(self):
        data = pd.read_pickle(self.filename)
        
        self.X = list(data['prob'])
        self.y = data["profession_class"].astype(np.int) #Verb
        self.gender_label = data["gender_class"].astype(np.int) # Gender
