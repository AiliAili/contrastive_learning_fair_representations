import pickle
import numpy as np
import pandas as pd

def load_dataset(path):
    
    with open(path, "rb") as f:
        
        data = pickle.load(f)
    return data

def load_dictionary(path):
    
    with open(path, "r", encoding = "utf-8") as f:
        
        lines = f.readlines()
        
    k2v, v2k = {}, {}
    for line in lines:
        
        k,v = line.strip().split("\t")
        v = int(v)
        k2v[k] = v
        v2k[v] = k
    
    return k2v, v2k    

if __name__ == "__main__":

    p2i, i2p = load_dictionary('/data/scratch/projects/punim0478/ailis/nullspace_projection-master/data/biasbios/profession2index.txt')
    g2i, i2g = load_dictionary('/data/scratch/projects/punim0478/ailis/nullspace_projection-master/data/biasbios/gender2index.txt')

    with open('/data/scratch/projects/punim0478/ailis/nullspace_projection-master/data/biasbios/train.pickle', 'rb') as f:
        train_data = pickle.load(f)
    train_data_frame = pd.DataFrame(train_data)
    train_data_cls_ny_1 = np.load('/data/scratch/projects/punim0478/ailis/nullspace_projection-master/data/bert_encode_biasbios/train_cls_1.npy')
    train_data_cls_ny_2 = np.load('/data/scratch/projects/punim0478/ailis/nullspace_projection-master/data/bert_encode_biasbios/train_cls_2.npy')
    train_data_cls_ny = np.concatenate((train_data_cls_ny_1, train_data_cls_ny_2), axis=0)
    train_data_avg_ny_1 = np.load('/data/scratch/projects/punim0478/ailis/nullspace_projection-master/data/bert_encode_biasbios/train_avg_1.npy')
    train_data_avg_ny_2 = np.load('/data/scratch/projects/punim0478/ailis/nullspace_projection-master/data/bert_encode_biasbios/train_avg_2.npy')
    train_data_avg_ny = np.concatenate((train_data_avg_ny_1, train_data_avg_ny_2), axis=0)

    assert len(train_data_frame) == len(train_data_cls_ny)
    assert len(train_data_cls_ny) == len(train_data_avg_ny)
    print('training size', len(train_data_avg_ny))

    train_data_frame['train_cls_data'] = list(train_data_cls_ny)
    train_data_frame['train_avg_data'] = list(train_data_avg_ny)

    train_main_label = [p2i[p] for p in train_data_frame["p"]]
    train_data_frame["profession_class"] = train_main_label

    train_gender_label = [g2i[g] for g in train_data_frame["g"]]
    train_data_frame["gender_class"] = train_gender_label

    train_data_frame.to_pickle("/data/scratch/projects/punim0478/ailis/nullspace_projection-master/data/biasbios/bios_train_df.pkl")

    with open('/data/scratch/projects/punim0478/ailis/nullspace_projection-master/data/biasbios/dev.pickle', 'rb') as f:
        dev_data = pickle.load(f)
    dev_data_frame = pd.DataFrame(dev_data)
    dev_data_cls_ny = np.load('/data/scratch/projects/punim0478/ailis/nullspace_projection-master/data/bert_encode_biasbios/dev_cls.npy')
    dev_data_avg_ny = np.load('/data/scratch/projects/punim0478/ailis/nullspace_projection-master/data/bert_encode_biasbios/dev_avg.npy')

    assert len(dev_data_frame) == len(dev_data_cls_ny)
    assert len(dev_data_cls_ny) == len(dev_data_avg_ny)
    print('dev size', len(dev_data_avg_ny))

    dev_data_frame['train_cls_data'] = list(dev_data_cls_ny)
    dev_data_frame['train_avg_data'] = list(dev_data_avg_ny)

    dev_main_label = [p2i[p] for p in dev_data_frame["p"]]
    dev_data_frame["profession_class"] = dev_main_label

    dev_gender_label = [g2i[g] for g in dev_data_frame["g"]]
    dev_data_frame["gender_class"] = dev_gender_label

    dev_data_frame.to_pickle("/data/scratch/projects/punim0478/ailis/nullspace_projection-master/data/biasbios/bios_dev_df.pkl")

    with open('/data/scratch/projects/punim0478/ailis/nullspace_projection-master/data/biasbios/test.pickle', 'rb') as f:
        test_data = pickle.load(f)
    test_data_frame = pd.DataFrame(test_data)
    test_data_cls_ny = np.load('/data/scratch/projects/punim0478/ailis/nullspace_projection-master/data/bert_encode_biasbios/test_cls.npy')
    test_data_avg_ny = np.load('/data/scratch/projects/punim0478/ailis/nullspace_projection-master/data/bert_encode_biasbios/test_avg.npy')

    assert len(test_data_frame) == len(test_data_cls_ny)
    assert len(test_data_cls_ny) == len(test_data_avg_ny)
    print('test size', len(test_data_avg_ny))

    test_data_frame['train_cls_data'] = list(test_data_cls_ny)
    test_data_frame['train_avg_data'] = list(test_data_avg_ny)

    test_main_label = [p2i[p] for p in test_data_frame["p"]]
    test_data_frame["profession_class"] = test_main_label

    test_gender_label = [g2i[g] for g in test_data_frame["g"]]
    test_data_frame["gender_class"] = test_gender_label

    test_data_frame.to_pickle("/data/scratch/projects/punim0478/ailis/nullspace_projection-master/data/biasbios/bios_test_df.pkl")




