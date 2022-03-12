import logging
from typing import Dict
import math
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.utils.data as data
from collections import Counter, defaultdict

def full_label_data(df, tasks):
    selected_rows = np.array([True]*len(df))
    for task in tasks:
        selected_rows = selected_rows & df[task].notnull().to_numpy()
    return selected_rows

class HateSpeechDataset(torch.utils.data.Dataset):
    def __init__(self, 
                args, 
                data_dir, 
                split, 
                balanced,
                balance_type,
                shuffle=None, 
                weight_scheme = None, 
                full_label_instances = False,
                private_label = "age",
                upsampling = False,
                embedding_type = "text_hidden",
                size = None,
                subsampling = False,
                subsampling_ratio = 1
                ):
        self.args = args
        self.data_dir = Path(data_dir)
        self.dataset_type = {"train", "valid", "test"}
        # check split
        assert split in self.dataset_type, "split should be one of train, valid, and test"
        self.split = split
        
        self.embedding_type = embedding_type

        # Load preprocessed tweets, labels, and tweet ids.
        print("Loading preprocessed Encoded data")
        if not full_label_instances:
            df, text_embedding, label, author_private_label_columns, total_n, selected_n= self.load_dataset(not_nan_col = [],
                                                                                                            author_private_labels = [private_label])
        else:
            df, text_embedding, label, author_private_label_columns, total_n, selected_n = self.load_dataset(not_nan_col = [private_label],
                                                                                                            author_private_labels = [private_label])

        self.X = text_embedding
        self.y = label
        self.private_label = author_private_label_columns[private_label]

        self.X = np.array(self.X)
        self.y = np.array(self.y)
        self.private_label = np.array(self.private_label)
        self.weight = []
        # Upsampling
        if full_label_instances and upsampling:
            if size == None:
                # Keep using the same size by default
                pass
            else:
                # Use the target size
                total_n = size
            
            # upsampling with replacement to get a same size dataset
            selected_indices = np.random.choice([i for i in range(selected_n)], replace=True, size=total_n)

            # update values
            self.X = self.X[selected_indices]
            self.y = self.y[selected_indices]
            self.private_label = self.private_label[selected_indices]
        elif full_label_instances and subsampling:
            # keep the same total number
            # reduce the number of distinct instances with private labels
            # 0 <= subsampling_ratio <= 1
            sub_indices = np.random.choice([i for i in range(selected_n)], replace=False, size = int(subsampling_ratio*selected_n))
            print("Number of distinct instances: {}".format(len(sub_indices)))
            
            selected_indices = np.random.choice(sub_indices, replace=True, size=selected_n)
            
            # update values
            self.X = self.X[selected_indices]
            self.y = self.y[selected_indices]
            self.private_label = self.private_label[selected_indices]
        else:
            pass


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
                group_labels = [(i,j) for i,j in zip(self.y, self.private_label)]
            elif balance_type == "y":
                group_labels = self.y
            elif balance_type == "g":
                group_labels = self.private_label
            elif balance_type == "class":
                group_labels = [(i,j) for i,j in zip(self.y, self.private_label)]
            elif balance_type == "stratified":
                group_labels = [(i,j) for i,j in zip(self.y, self.private_label)]
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
                    y_group_idx = [group_idx[(y, g)] for g in set(self.private_label)]
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
                distinct_g_label = set(self.private_label)
                
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
            private_label = [self.private_label[index] for index in selected_index]
            self.private_label = private_label
        elif balanced == False:
            group_idx = {}
            inverse_weights = {}
            if weight_scheme == 'joint':
                group_labels = [(i,j) for i,j in zip(self.y, self.private_label)]
                for idx, group_label in enumerate(group_labels):
                    group_idx[group_label] = group_idx.get(group_label, []) + [idx]

                for tem in group_idx:
                    inverse_weights[tem] = len(self.y)/float(len(group_idx[tem]))
                
                for i in range(0, len(self.y)):
                    self.weight.append(inverse_weights[(self.y[i], self.private_label[i])])
            
            else:
                for i in range(0, len(self.y)):
                    self.weight.append(1.0)

        self.X = np.array(self.X)
        self.y = np.array(self.y)
        self.private_label = np.array(self.private_label)
        if len(self.weight) == 0:
            self.weight = [1]*len(self.y)
        #self.location_label = np.array(self.location_label)
 
        group_labels = [(i,j) for i,j in zip(self.y, self.private_label)]
        print(Counter(group_labels))

        print("Done, loaded data shapes: {}, {}, {}".format(self.X.shape, self.y.shape, self.private_label.shape))
        #print(Counter(self.y))
        #print(Counter(self.private_label))
        #freq_dict = defaultdict(int) 
        #for i in range(0, len(self.y)):
        #    freq_dict[(self.y[i], self.private_label[i])]+=1
        #print(freq_dict)

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.y)

    def __getitem__(self, index):
        'Generates one sample of data'
        if math.isnan(self.private_label[index]):
            return self.X[index], self.y[index], 2
        return self.X[index], self.y[index], self.private_label[index], self.weight[index]

    def load_dataset(
        self,
        not_nan_col = [],
        author_private_labels = ["age"]
        ):
        dataset_dir = self.data_dir / "{}_set.pkl".format(self.split)
        df = pd.read_pickle(dataset_dir)
        # df = df.replace("x", np.nan)

        total_n = len(df)

        df = df[full_label_data(df, not_nan_col)]

        selected_n = len(df)

        print("Select {} out of {} in total.".format(selected_n, total_n))

        # input and labels
        text_embedding = list(df[self.embedding_type])
        label = list(df["label"])
        # private_labels
        author_private_label_columns = {
            p_name:list(df[p_name].astype("float"))
            for p_name in author_private_labels
        }

        for p_name in author_private_labels:
            df[p_name] = author_private_label_columns[p_name]

        return df, text_embedding, label, author_private_label_columns, total_n, selected_n


if __name__ == "__main__":
    class Args:
        gender_balanced = False
    
    # data_path = "D:\\Project\\User_gender_removal\\data\\deepmoji\\"
    data_path = "D:\\Project\\adv_decorrelation\\data\\hate_speech"

    split = "train"
    args = Args()
    my_dataset = HateSpeechDataset(args, 
                                data_path, 
                                split, 
                                full_label_instances = True, 
                                upsampling = False,
                                private_label = "age",
                                embedding_type = "deepmoji_encoding",
                                size=None,
                                subsampling = True,
                                subsampling_ratio = 0.5
                                )