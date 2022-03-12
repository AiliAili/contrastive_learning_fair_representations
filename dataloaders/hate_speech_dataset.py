import numpy as np
import pandas as pd
from pathlib import Path

from typing import Dict

import torch
import torch.utils.data as data


def full_label_data(df, tasks):
    selected_rows = np.array([True]*len(df))
    for task in tasks:
        selected_rows = selected_rows & df[task].notnull().to_numpy()
    #print('hello world', sum(selected_rows))
    return selected_rows

def load_dataset(
    folder_dir,
    split,
    not_nan_col = ["age", "ethnicity"],
    author_private_labels = ["age", "ethnicity"],
    target_private_labels = ["target_gender"]
    ):
    assert split in ["train", "valid", "test"], "split is invalid"
    dataset_dir = folder_dir + "{}_set.pkl".format(split)
    df = pd.read_pickle(dataset_dir)
    df = df.replace("x", np.nan)

    df = df[full_label_data(df, not_nan_col)]

    # input and labels
    text_embedding = list(df["text_hidden"])
    label = list(df["label"])
    # private_labels
    author_private_label_columns = {
        p_name:list(df[p_name].astype("float"))
        for p_name in author_private_labels
    }

    for p_name in author_private_labels:
        df[p_name] = author_private_label_columns[p_name]

    cat_private_labels = np.concatenate(
        [
            np.array(author_private_label_columns[p_name]).reshape(-1,1) for p_name in author_private_labels
        ],
        axis = 1
    )

    return df, text_embedding, label, author_private_label_columns, cat_private_labels

class HateSpeechDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, split, not_nan_col = ["gender", "age", "country", "ethnicity"]):

        self.data_dir = Path(data_dir)
        self.dataset_type = {"train", "valid", "test"}
        # check split
        assert split in self.dataset_type, "split should be one of train, valid, and test"
        self.split = split

        # Load preprocessed tweets, labels, and tweet ids.
        print("Loading preprocessed Encoded data")
        df, text_embedding, label, author_private_label_columns = self.load_dataset(
            not_nan_col = not_nan_col,
            author_private_labels = ["gender", "age", "country", "ethnicity"],
            )

        self.X = text_embedding
        self.y = label
        self.gender_label = author_private_label_columns["gender"]
        self.age_label = author_private_label_columns["age"]
        self.country_label = author_private_label_columns["country"]
        self.ethnicity_label = author_private_label_columns["ethnicity"]

        self.X = np.array(self.X)
        self.y = np.array(self.y)
        self.gender_label = np.array(self.gender_label)
        self.age_label = np.array(self.age_label)
        self.country_label = np.array(self.country_label)
        self.ethnicity_label = np.array(self.ethnicity_label)

        print("Done, loaded data shapes: {}, {}, {}".format(self.X.shape, self.y.shape, self.gender_label.shape))

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.y)

    def __getitem__(self, index):
        'Generates one sample of data'
        return self.X[index], self.y[index], self.gender_label[index], self.age_label[index], self.country_label[index], self.ethnicity_label[index]

    def load_dataset(
        self,
        not_nan_col = [],
        author_private_labels = []
        ):
        
        dataset_dir = self.data_dir / "{}_set.pkl".format(self.split)
        df = pd.read_pickle(dataset_dir)
        #print(not_nan_col, len(df))

        df = df[full_label_data(df, not_nan_col)]

        # input and labels
        text_embedding = list(df["text_hidden"])
        label = list(df["label"])
        # private_labels
        author_private_label_columns = {
            p_name:list(df[p_name].astype("float"))
            for p_name in author_private_labels
        }

        for p_name in author_private_labels:
            df[p_name] = author_private_label_columns[p_name]

        return df, text_embedding, label, author_private_label_columns


if __name__ == "__main__":
    folder_dir = "D:\\Project\\joint_debiasing\\data\\hate_speech\\"
    train_path = "train_set.pkl"
    dev_path = "valid_set.pkl"
    test_path = "test_set.pkl"

    (
    train_df,
    train_text_embd, 
    train_label, 
    author_private_label_columns,
    train_cat_private_labels) = load_dataset(folder_dir, 
                                            "train")
    print(np.array(train_text_embd).shape)
    print(train_cat_private_labels.shape)

    print(type(train_cat_private_labels[0][1]))

    from collections import Counter
    print(Counter(author_private_label_columns["age"]))
    print((author_private_label_columns["age"][:5]))
    

    split = "train"
    my_dataset = HateSpeechDataset(folder_dir, split)
    
    from collections import Counter
    print(max(np.array(list(Counter(my_dataset.age_label)))))