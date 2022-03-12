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
    return selected_rows

class BiosDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 data_dir, 
                 split, 
                 embedding_type = "bert_avg_SE",
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
        self.location_label = []

        # Load preprocessed data
        print("Loading data")
        self.load_data()

        self.X = np.array(self.X)
        if len(self.X.shape) == 3:
            self.X = np.concatenate(list(self.X), axis=0)
        self.y = np.array(self.y)
        self.gender_label = np.array(self.gender_label)
        self.location_label = np.array(self.location_label)
        
        print("Done, loaded data shapes: {}, {}".format(self.X.shape, self.y.shape))

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.y)

    def __getitem__(self, index):
        'Generates one sample of data'
        return self.X[index], self.y[index], self.gender_label[index]#self.location_label[index], 
    
    def load_data(self):
        data = pd.read_pickle(self.data_dir+'/'+self.filename)
 
        self.X = list(data[self.embedding_type])
        self.y = data["profession_class"].astype(np.float) #Profession
        self.location_label = data["econ_class"].astype(np.int) # location
        self.gender_label = data["gender_class"].astype(np.int) # Gender

        return 0

def load_dataset(
    folder_dir,
    split,
    author_private_labels = ["econ_class", "gender_class"],
    embedding_type = "bert_avg_SE"
    ):

    assert split in ["train", "dev", "test"], "split is invalid"
    dataset_dir = folder_dir / "bios_{}_df.pkl".format(split)
    df = pd.read_pickle(dataset_dir)

    # check embedding type
    assert embedding_type in ("avg", "cls", "bert_avg_SE"), "Embedding should either be avg or cls."
    """
        avg and cls are sentence representations from BERT encoders
        bert_avg_SE are sentecne representations from fine-tuned BERT encoders
    """
    if embedding_type in ("avg", "cls"):
        embedding_type = "train_{}_data".format(embedding_type)

    # input and labels
    text_embedding = np.concatenate(list(df[embedding_type]), axis=0)
    label = list(df["profession_class"])
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


if __name__ == "__main__":
    folder_dir = Path("/home/xudongh1/Project/joint_debiasing/data/bios")

    split = "train"
    my_dataset = BiosDataset(folder_dir, split)
    
    from collections import Counter
    print(Counter(my_dataset.gender_label))
    print(Counter(my_dataset.location_label))
    print(Counter(my_dataset.y))
    print(len(Counter(my_dataset.y)))
    print(max(np.array(list(Counter(my_dataset.gender_label)))))
