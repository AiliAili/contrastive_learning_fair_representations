import os,argparse,time
import numpy as np
from datetime import datetime
import random
import torch
import torch.backends.cudnn as cudnn
import torch.optim 
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.data.sampler import WeightedRandomSampler
import torch.nn.functional as F
import time

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

from dataloaders.verb_dataset import VerbDataset
from dataloaders.scheduler import BalancedBatchSampler
from networks.deepmoji_sa import DeepMojiModel
from networks.discriminator import Discriminator


from tqdm import tqdm, trange
from networks.customized_loss import DiffLoss
from networks.contrastive_loss import Contrastive_Loss

from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from networks.eval_metrices_updated import leakage_evaluation, tpr_multi, leakage_hidden, leakage_logits

from pathlib import Path, PureWindowsPath
from collections import defaultdict, Counter
import pickle
from random import shuffle
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr
import argparse

# train a discriminator 1 epoch
def adv_train_epoch(model, discriminators, iterator, adv_optimizers, criterion, device, args):
    """"
    Train the discriminator to get a meaningful gradient
    """

    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    for discriminator in discriminators:
        discriminator.train()

    # deactivate gradient reversal layer
    for discriminator in discriminators:
        discriminator.GR = False
    
    for batch in iterator:
        
        text = batch[0].float()
        tags = batch[1].long()
        p_tags = batch[2].long()

        text = text.to(device)
        tags = tags.to(device)
        p_tags = p_tags.to(device)
        
        hs = model.hidden(text).detach()
        
        # iterate all discriminators
        for discriminator, adv_optimizer in zip(discriminators, adv_optimizers):
        
            adv_optimizer.zero_grad()

            adv_predictions = discriminator(hs)

        
            loss = criterion(adv_predictions, p_tags)

            # encrouge orthogonality
            if args.DL == True:
                # Get hidden representation.
                adv_hs_current = discriminator.hidden_representation(hs)
                for discriminator2 in discriminators:
                    if discriminator != discriminator2:
                        adv_hs = discriminator2.hidden_representation(hs)
                        # Calculate diff_loss
                        # should not include the current model
                        difference_loss = args.diff_LAMBDA * args.diff_loss(adv_hs_current, adv_hs)
                        loss = loss + difference_loss
                        
            loss.backward()
        
            adv_optimizer.step()
            epoch_loss += loss.item()

    return epoch_loss / len(iterator)

# evaluate the discriminator
def adv_eval_epoch(model, discriminators, iterator, criterion, device, args):

    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    for discriminator in discriminators:
        discriminator.eval()

    # deactivate gradient reversal layer
    for discriminator in discriminators:
        discriminator.GR = False
    

    preds = {i:[] for i in range(args.n_discriminator)}
    labels = []
    private_labels = []

    for batch in iterator:
        
        text = batch[0]
        tags = batch[1]
        p_tags = batch[2]

        text = text.to(device).float()
        tags = tags.to(device).long()
        p_tags = p_tags.to(device).long()
        
        # extract hidden state from the main model
        hs = model.hidden(text)
        # let discriminator make predictions

        for index, discriminator in enumerate(discriminators):
            adv_pred = discriminator(hs)
        
            loss = criterion(adv_pred, p_tags)
                        
            epoch_loss += loss.item()
        
            adv_predictions = adv_pred.detach().cpu()
            preds[index] += list(torch.argmax(adv_predictions, axis=1).numpy())


        tags = tags.cpu().numpy()

        labels += list(tags)
        
        private_labels += list(batch[2].cpu().numpy())
        
    
    return ((epoch_loss / len(iterator)), preds, labels, private_labels)

# train the main model with adv loss
def train_epoch(model, discriminators, iterator, optimizer, criterion, contrastive_loss, device, args):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.train()
    for discriminator in discriminators:
        discriminator.train()

    # activate gradient reversal layer
    for discriminator in discriminators:
        discriminator.GR = True
    
    for batch in iterator:
        
        text = batch[0].float()
        tags = batch[1].long()
        p_tags = batch[2].long()
        #print(tags)

        text = text.to(device)
        tags = tags.to(device)
        p_tags = p_tags.to(device)
        #print(Counter(tags.cpu().numpy()))
        optimizer.zero_grad()
        # main model predictions
        predictions, features_1, features_2, _ = model(text)

        if args.loss_type == 'ce':
            loss = criterion(predictions, tags) 
            '''for i in range(0, 211):
                indices_0 = set(torch.where(tags == i)[0].cpu().numpy()).intersection(set(torch.where(p_tags == 0)[0].cpu().numpy()))
                indices_0 = list(indices_0)
                indices_1 = set(torch.where(tags == i)[0].cpu().numpy()).intersection(set(torch.where(p_tags == 1)[0].cpu().numpy()))
                indices_1 = list(indices_1)
                if len(indices_0) > 0 and len(indices_1) > 0:
                    #print(len(indices_0), len(indices_1))
                    #loss+=0.1*criterion(predictions[indices_0], tags[indices_0])/criterion(predictions[indices_1], tags[indices_1])
                    #loss+=0.1*criterion(predictions[indices_1], tags[indices_1])/criterion(predictions[indices_0], tags[indices_0]) 
                    tem_0 = criterion(predictions[indices_0], tags[indices_0])
                    tem_1 = criterion(predictions[indices_1], tags[indices_1])
                    #loss+= 0.5*abs(tem_0-tem_1)
                    if tem_0 > tem_1:
                        loss+=0.04*(tem_0-tem_1)
                    else:
                        loss+=0.04*(tem_1-tem_0)'''

        elif args.loss_type == 'scl1':
            loss = contrastive_loss(features_2, tags)
        elif args.loss_type == 'scl2':
            loss = contrastive_loss(features_2, p_tags)
        elif args.loss_type == 'ce+scl1':
            loss = criterion(predictions, tags)
            con_loss = contrastive_loss(features_2, tags)
            loss*=args.lambda_weight
            loss+=(1-args.lambda_weight)*con_loss
            
        elif args.loss_type == 'ce-scl2':
            loss = criterion(predictions, tags)
            protected_loss = contrastive_loss(features_2, p_tags)
            loss*=args.lambda_weight
            loss-=(1-args.lambda_weight)*protected_loss
            
            '''for i in range(0, 211):
                indices = torch.where(tags == i)[0].cpu().numpy()
                #print(Counter(p_tags[indices].cpu().numpy()))
                if len(indices) > 0 and len(Counter(p_tags[indices].cpu().numpy()))==2:              
                    protected_loss = contrastive_loss(features_2[indices], p_tags[indices])                  
                    loss-=(1-args.lambda_weight)*protected_loss''' 
        elif args.loss_type == 'scl1-scl2':
            loss = criterion(predictions, tags)
            con_loss = contrastive_loss(features_2, tags)
            protected_loss = contrastive_loss(features_2, p_tags)
            loss = (1-args.lambda_weight)*con_loss-(1-args.lambda_weight)*protected_loss
        elif args.loss_type == 'ce+scl1-scl2':
            loss = criterion(predictions, tags)
            con_loss = contrastive_loss(features_2, tags)
            protected_loss = contrastive_loss(features_2, p_tags)
            #print(con_loss, protected_loss)
            loss*=args.lambda_weight
            for i in range(0, 211):
                indices = torch.where(tags == i)[0].cpu().numpy()
                #print(Counter(p_tags[indices].cpu().numpy()))
                if len(indices) > 0 and len(Counter(p_tags[indices].cpu().numpy()))==2:              
                    protected_loss = contrastive_loss(features_2[indices], p_tags[indices])                   
                    loss-=(1-args.lambda_weight)*protected_loss

            
            loss+=(1-args.lambda_weight)*con_loss
            #loss-=(1-args.lambda_weight)*protected_loss
            #print(con_loss, protected_loss)

        if args.adv:
            # discriminator predictions
            p_tags = p_tags.long()

            hs = model.hidden(text)

            for discriminator in discriminators:
                adv_predictions = discriminator(hs)
            
                loss = loss + (criterion(adv_predictions, p_tags) / len(discriminators))
                        
        loss.backward()

        optimizer.step()
        epoch_loss += loss.item()
        
        
    return epoch_loss / len(iterator)

# to evaluate the main model
def eval_main(model, iterator, criterion, device, args):
    
    epoch_loss = 0
    
    model.eval()
    
    preds = []
    labels = []
    private_labels = []

    for batch in iterator:
        
        text = batch[0]

        tags = batch[1]
        # tags = batch[2] #Reverse
        p_tags = batch[2]
        
        text = text.to(device).float()
        tags = tags.to(device).long()
        #p_tags = p_tags.to(device).float()
        p_tags = p_tags.to(device).long()
        
        #text = F.normalize(text, dim=1)
        predictions, _, _, _ = model(text)
        
        loss = criterion(predictions, tags)
        
        epoch_loss += loss.item()
        
        predictions = predictions.detach().cpu()
        tags = tags.cpu().numpy()

        preds += list(torch.argmax(predictions, axis=1).numpy())
        labels += list(tags)

        private_labels += list(batch[2].cpu().numpy())
    
    return ((epoch_loss / len(iterator)), preds, labels, private_labels)

def get_leakage_data(model, iterator, filename, device, args):
    model.eval()
    
    data_frame = pd.DataFrame()
    preds = []
    labels = []
    private_labels = []
    image_list = []
    second_last_representation = []
    tem_preds = []
    for batch in iterator:
        
        text = batch[0].float()
        tags = batch[1].long()
        
        text = text.to(device)
        tags = tags.to(device)
        
        #text = F.normalize(text, dim=1)
        predictions, _, _, second_last = model(text)
        
        predictions = predictions.detach().cpu()
        preds+=predictions.tolist()

        second_last = second_last.detach().cpu()
        second_last_representation+=second_last.tolist()

        tem_preds += list(torch.argmax(predictions, axis=1).numpy())

        labels +=tags.cpu().tolist()
        private_labels += list(batch[2].tolist())
        image_list+=list(batch[3])
    
    data_frame['prob'] = preds
    data_frame['verb'] = labels
    data_frame['gender'] = private_labels
    data_frame['image_name'] = image_list
    data_frame['second_last_representation'] = second_last_representation
    data_frame['predict'] = tem_preds
    data_frame.to_pickle(filename)
    accuracy = accuracy_score(labels, tem_preds)
    print('Potential', accuracy)

    X_logits = list(data_frame['prob'])
    X_hidden = list(data_frame['second_last_representation'])
    y = list(data_frame['verb'])
    gender_label = list(data_frame['gender'])

    return (X_logits, X_hidden, y, gender_label)

def load_leakage_data(filename):
    data = pd.read_pickle(filename)
    X_logits = list(data['prob'])
    X_hidden = list(data['second_last_representation'])
    y = list(data['verb'])
    gender_label = list(data['gender'])
        
    return (X_logits, X_hidden, y, gender_label)

def get_group_metrics(preds, labels, p_labels):
    preds_0 = []
    labels_0 = []
    preds_1 = []
    labels_1 = []
    for i in range(0, len(p_labels)):
        if p_labels[i] == 0:
            preds_0.append(preds[i])
            labels_0.append(labels[i])
        else:
            preds_1.append(preds[i])
            labels_1.append(labels[i])

    accuracy_0 = 100*accuracy_score(labels_0, preds_0)
    accuracy_1 = 100*accuracy_score(labels_1, preds_1)
    if len(preds_0) <= len(preds_1):
        minority = accuracy_0
    else:
        minority = accuracy_1

    print(len(preds_0), len(preds_1), accuracy_0, accuracy_1)
    return abs(accuracy_0-accuracy_1), min(accuracy_0, accuracy_1), 0.5*(accuracy_0+accuracy_1), minority

def log_uniform(power_low, power_high):
    return np.power(10, np.random.uniform(power_low, power_high))

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def get_TPR(y_pred, y_true, i2p, i2g, gender, counter):
    
    scores = defaultdict(Counter)
    prof_count_total = defaultdict(Counter)
    
    for y_hat, y, g in zip(y_pred, y_true, gender):
        
        if y == y_hat:
            
            scores[i2p[y]][i2g[g]] += 1
        
        prof_count_total[i2p[y]][i2g[g]] += 1
    #print(scores)
    #print(prof_count_total)
    tprs = defaultdict(dict)
    tprs_change = dict()
    tprs_ratio = []
    
    for profession, scores_dict in scores.items():
        
        good_m, good_f = scores_dict["m"], scores_dict["f"]
        prof_total_f = prof_count_total[profession]["f"]
        prof_total_m = prof_count_total[profession]["m"]
        
        #if profession not in ['Cause_motion', 'Resolve_problem', 'Attaching']:# in ['Manipulation', 'Placing', 'Self_motion', 'Body_movement']:
        if True:
            #print(profession, scores_dict, prof_total_m, prof_total_f)
            tpr_m = 100*(good_m) / prof_total_m
            tpr_f = 100*(good_f) / prof_total_f
            
            tprs[profession]["m"] = tpr_m
            tprs[profession]["f"] = tpr_f
            tprs_ratio.append(0)
            tprs_change[profession] = tpr_f - tpr_m
            #print(profession, abs(tpr_f - tpr_m))
    
    value = []
    weighted_value = []
    for profession in tprs_change:
        value.append(tprs_change[profession]**2)
        weighted_value.append(counter[profession]*(tprs_change[profession]**2))

    #return tprs, tprs_change, np.mean(np.abs(tprs_ratio)) 
    return np.sqrt(np.mean(value)), np.sqrt(np.mean(weighted_value)), tprs_change

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

def woman_profession_portion(train_data, dev_data, test_data, i2p, i2g):
    counter = defaultdict(Counter)
    for i in range(0, len(train_data.y)):
        profession = i2p[train_data.y[i]]
        gender = i2g[train_data.gender_label[i]]
        counter[profession][gender]+=1

    for i in range(0, len(dev_data.y)):
        profession = i2p[dev_data.y[i]]
        gender = i2g[dev_data.gender_label[i]]
        counter[profession][gender]+=1

    for i in range(0, len(test_data.y)):
        profession = i2p[test_data.y[i]]
        gender = i2g[test_data.gender_label[i]]
        counter[profession][gender]+=1
    
    prof2fem = dict()
    for k, values in counter.items():
        prof2fem[k] = values['f']/float((values['f']+values['m']))

    return prof2fem


def correlation_plot(tprs, prof2fem, filename):
    professions = list(tprs.keys())
    tpr_lst = [tprs[p] for p in professions]
    proportion_lst = [prof2fem[p] for p in professions]
    plt.plot(proportion_lst, tpr_lst, marker = "o", linestyle = "none")
    plt.xlabel("% women", fontsize = 13)
    plt.ylabel(r'$GAP_{female,y}^{TPR}$', fontsize = 13)

    for p in professions:
        x,y = prof2fem[p], tprs[p]
        plt.annotate(p , (x,y), size = 7, color = "red")
    
    #plt.ylim(-0.4, 0.55)
    z = np.polyfit(proportion_lst, tpr_lst, 1)
    p = np.poly1d(z)
    plt.plot(proportion_lst,p(proportion_lst),"r--")
    plt.savefig("./verb_tem.png", dpi = 600)
    print("Correlation: {}; p-value: {}".format(*pearsonr(proportion_lst, tpr_lst)))
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_fp16', action='store_true')
    parser.add_argument('--cuda', type=str)
    parser.add_argument('--hidden_size', type=int, default = 300)
    parser.add_argument('--emb_size', type=int, default = 2048)
    parser.add_argument('--num_classes', type=int, default = 211)
    parser.add_argument('--adv', action='store_true')
    parser.add_argument('--adv_level', type=int, default = -1)
    parser.add_argument('--lr', type=float, default=0.00003)
    parser.add_argument('--starting_power', type=int)
    parser.add_argument('--LAMBDA', type=float, default=0.8)
    parser.add_argument('--n_discriminator', type=int, default = 0)
    parser.add_argument('--adv_units', type=int, default = 256)
    parser.add_argument('--ratio', type=float, default=0.8)
    parser.add_argument('--DL', action='store_true')
    parser.add_argument('--diff_LAMBDA', type=float, default=1000)
    parser.add_argument('--data_path', type=str, default='/data/scratch/projects/punim0478/ailis/MSCOCO/Balanced-Datasets-Are-Not-Enough-master/verb_classification/data/', help='directory containing the dataset')
    parser.add_argument('--temperature', type=float, default=0.01)
    parser.add_argument('--batch_size', type=int, default = 1024)    
    parser.add_argument('--seed', type=int, default = 46)
    parser.add_argument('--representation_file', default='./analysis/ce+scl1+scl2.txt', type=str, help='the file storing test representation before the classifier layer')
    parser.add_argument('--loss_type', default='ce', type=str, help='the type of loss we want to use')
    parser.add_argument('--lambda_weight', type=float, default=0.5, help='the weight for supervised contrastive learning over main labels')
    parser.add_argument('--lambda_1', type=float, default=0.5, help='the weight for supervised contrastive learning over main labels')
    parser.add_argument('--lambda_2', type=float, default=0.5, help='the weight for supervised contrastive learning over main labels')
    parser.add_argument('--num_epochs', type=int, default = 5)
    parser.add_argument('--device_id', type=int, default = 0)
    parser.add_argument('--patience', type=int, default = 5)
    parser.add_argument('--experiment_type', default='adv_experiment_lambda', type=str, help='which types of experiments we are doing')
    parser.add_argument('--gender_balanced', action='store_true')
    parser.add_argument('--whether_vanilla', action='store_true')

    args = parser.parse_args()

    accumulate_accuracy = []
    accumulate_rms_diff = []
    accumulate_weighted_rms_diff = []
    accumulate_f1_macro = []
    accumulate_f1_micro = []
    accumulate_f1_weighted = []

    verb_id_map = pickle.load(open(args.data_path+'verb_id.map', 'rb'))
    verb2id = verb_id_map['verb2id']
    id2verb = verb_id_map['id2verb']
    g2i, i2g = load_dictionary("./gender2index.txt")
    #print(id2verb)

    train_time_list = []
    train_epochs_list = []

    #batch_list = [256, 512, 1024, 2048, 4096]
    #lr_list = [1e-5, 3e-5, 5e-5, 7e-5, 1e-4, 3e-4, 5e-4, 7e-4, 1e-3, 3e-3, 5e-3, 7e-3, 1e-2]
    #lambda_ratio_list = [1e-2, 5e-2, 1e-1, 5e-1, 1e0, 5e0, 1e1, 5e1, 1e2]
    lambda_ratio_list = [5e-1]
    #lambda_1_list = [1e-2, 5e-2, 1e-1, 5e-1, 1e0, 5e0]
    #lambda_2_list = [1e-2, 5e-2, 1e-1, 5e-1, 1e0, 5e0]

    batch_list = [256, 512, 1024, 2048, 4096]
    lr_list = [1e-5, 3e-5, 5e-5, 7e-5, 1e-4, 3e-4, 5e-4, 7e-4, 1e-3, 3e-3, 5e-3, 7e-3, 1e-2]
    lambda_ratio_list = [1e-2, 5e-2, 1e-1, 5e-1, 1e0, 5e0, 1e1, 5e1, 1e2]


    adv_batch_list = [256, 512, 1024, 2048, 4096]
    adv_lr_list = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
    adv_lambda_list = [1e-2, 5e-2, 1e-1, 5e-1, 1e0]
    adv_diff_lambda_list = [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4]


    accumulate_acc = []
    accumulate_rms_diff = []
    accumulate_weighted_rms_diff = []
    accumulate_leakage_logits = []
    accumulate_leakage_hidden = []
    count_runs = 0
    output_file = open(args.representation_file, 'w')

    #for tem_batch in batch_list:
    #    for tem_lr in lr_list:
    #        for tem_lambda in lambda_ratio_list:
    '''if True:
        if True:
            if True:
                tem_lr = 0
                tem_batch = 0
                tem_lambda = 0'''
   
    #if True:
    #    if True:
    #        for tem_lambda in adv_diff_lambda_list:

    selected_lambda = args.lambda_weight
    #if True:
    #    if True:
    #       for tem_seed in [40, 41, 42, 43, 44, 45, 46, 47, 48, 49]:
    #        #for tem_lambda in adv_diff_lambda_list:
    '''for tem_batch in adv_batch_list:
        for tem_lr in adv_lr_list:
            for tem_lambda in adv_lambda_list:  '''    
    #for tem_batch in batch_list:
    #    for tem_lr in lr_list:
            #for tem_lambda in lambda_ratio_list: 
    if True:
        if True:
            for tem_seed in [40, 41, 42, 43, 44, 45, 46, 47, 48, 49]:    
                #args.LAMBDA = tem_lambda
                #args.lambda_weight = 1/float(1+tem_lambda)
                #args.lr = tem_lr
                #args.batch_size = tem_batch
                #args.lambda_weight = tem_batch
                #tem_lambda = tem_batch
            
                #args.diff_LAMBDA = tem_lambda
                #args.lambda_weight = tem_lambda

                args.lambda_weight = 1/float(1+selected_lambda)
                tem_lambda = tem_seed
                args.seed = tem_seed
                print('============================================================')
                seed_everything(args.seed)

                #args.lambda_weight = 1/float(1+tem_lambda)
                #args.lr = tem_lr
                #args.batch_size = tem_batch

                print('verb batch size', args.batch_size, 'lr', args.lr, 'lambda_weight', args.lambda_weight)
                #print('batch size', args.batch_size, 'lr', args.lr, 'lambda_weight', args.lambda_weight, args.lambda_1, args.lambda_2)
                # file names
                experiment_type = "adv_Diverse"
                
                # path to checkpoints
                main_model_path = "./official_models/verb_full_model_{}_{}_{}_{}_{}_{}.pt".format(args.experiment_type, args.lr, args.batch_size, args.loss_type, args.lambda_weight, args.seed)
                adv_model_path = "./official_models/verb_full_discriminator_{}_{}_{}_{}.pt"

                # Device
                device = torch.device("cuda:"+str(args.device_id))
                balance_type = 'g'
                # Load data
                train_data = VerbDataset(args.data_path, "full_train", shuffle, balanced=False, balance_type=balance_type, vanilla=args.whether_vanilla)
                dev_data = VerbDataset(args.data_path, "full_val", shuffle, balanced=False, balance_type=balance_type, vanilla=args.whether_vanilla)
                test_data = VerbDataset(args.data_path, "full_test", shuffle, balanced=False, balance_type=balance_type, vanilla=args.whether_vanilla)


                '''class_sample_count = np.array([len(np.where(train_data.y == t)[0]) for t in np.unique(train_data.y)])
                weight = 1. / class_sample_count
                samples_weight = np.array([weight[t] for t in train_data.y])

                samples_weight = torch.from_numpy(samples_weight)
                samples_weigth = samples_weight.double()
                sampler = WeightedRandomSampler(samples_weight, len(samples_weight))'''

                n_classes = 2
                n_samples = 5000

                #balanced_batch_sampler = BalancedBatchSampler(train_data, n_classes, n_samples)
                #training_generator = torch.utils.data.DataLoader(train_data, batch_sampler=balanced_batch_sampler)
                '''my_testiter = iter(dataloader)
                tem = my_testiter.next()
                print(tem[1])

                exit()'''
                
                prof2fem = woman_profession_portion(train_data, dev_data, test_data, id2verb, i2g)
                # Data loader
                training_generator = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=0)
                validation_generator = torch.utils.data.DataLoader(dev_data, batch_size=args.batch_size, shuffle=False, num_workers=0)
                test_generator = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=0)


                # Init model
                model = DeepMojiModel(args)
                '''for name, param in model.named_parameters():
                    if not name.startswith('dense3'):
                    #if param.requires_grad:
                        print(name)

                exit(0)'''
                model = model.to(device)

                # Init discriminators
                # Number of discriminators
                n_discriminator = args.n_discriminator

                discriminators = [Discriminator(args, args.hidden_size, 2) for _ in range(n_discriminator)]
                discriminators = [dis.to(device) for dis in discriminators]

                diff_loss = DiffLoss()
                args.diff_loss = diff_loss

                contrastive_loss = Contrastive_Loss(device=device, temperature=args.temperature, base_temperature= args.temperature)

                # Init optimizers
                LEARNING_RATE = args.lr
                optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)

                adv_optimizers = [Adam(filter(lambda p: p.requires_grad, dis.parameters()), lr=1e-1*LEARNING_RATE) for dis in discriminators]

                # Init learing rate scheduler
                scheduler = ReduceLROnPlateau(optimizer, mode = 'max', factor = 0.5, patience = 2)

                # Init criterion
                criterion = torch.nn.CrossEntropyLoss()
                
                best_loss, valid_preds, valid_labels, _ = eval_main(
                                                                    model = model, 
                                                                    iterator = validation_generator, 
                                                                    criterion = criterion, 
                                                                    device = device, 
                                                                    args = args
                                                                    )
                best_loss = float('inf')
                best_acc = accuracy_score(valid_labels, valid_preds)
                best_epoch = 0

                before_train = time.time()

                for i in trange(args.num_epochs):
                    '''if i %2 == 0:
                        args.loss_type = 'ce'
                        #for name, param in model.named_parameters():
                        #    if not name.startswith('dense3'):
                        #        param.requires_grad = False

                    else:
                        args.loss_type = 'ce+scl1-scl2'''
                    train_epoch(
                                model = model, 
                                discriminators = discriminators, 
                                iterator = training_generator, 
                                optimizer = optimizer, 
                                criterion = criterion, 
                                contrastive_loss = contrastive_loss,
                                device = device, 
                                args = args
                                )

                    valid_loss, valid_preds, valid_labels, _ = eval_main(
                                                                        model = model, 
                                                                        iterator = validation_generator, 
                                                                        criterion = criterion, 
                                                                        device = device, 
                                                                        args = args
                                                                        )
                    valid_acc = accuracy_score(valid_labels, valid_preds)
                    #print(valid_acc)
                    # learning rate scheduler
                    scheduler.step(valid_loss)
                    #print('Valid loss', valid_loss, 'Valid acc', valid_acc, best_epoch, i, args.loss_type)

                    #test_loss, preds, labels, p_labels = eval_main(model, test_generator, criterion, device, args)
                    #accuracy = accuracy_score(labels, preds)
                    #print('overall accuracy', accuracy)

                    #rms_diff = get_TPR(preds, labels, i2p, i2g, p_labels)
                    #print('rms diff', rms_diff)
                    # early stopping
                    if valid_loss < best_loss:
                        best_acc = valid_acc
                        best_loss = valid_loss
                        best_epoch = i
                        torch.save(model.state_dict(), main_model_path)
                    else:
                        if best_epoch+args.patience<=i:
                            break

                    # Train discriminator untile converged
                    # evaluate discriminator 
                    best_adv_loss, _, _, _ = adv_eval_epoch(
                                                            model = model, 
                                                            discriminators = discriminators, 
                                                            iterator = validation_generator, 
                                                            criterion = criterion, 
                                                            device = device, 
                                                            args = args
                                                            )
                    best_adv_epoch = -1
                    for k in range(100):
                        adv_train_epoch(
                                        model = model, 
                                        discriminators = discriminators, 
                                        iterator = training_generator, 
                                        adv_optimizers = adv_optimizers, 
                                        criterion = criterion, 
                                        device = device, 
                                        args = args
                                        )
                        adv_valid_loss, _, _, _ = adv_eval_epoch(
                                                                model = model, 
                                                                discriminators = discriminators, 
                                                                iterator = validation_generator, 
                                                                criterion = criterion, 
                                                                device = device, 
                                                                args = args
                                                                )
                            
                        if adv_valid_loss < best_adv_loss:
                                best_adv_loss = adv_valid_loss
                                best_adv_epoch = k
                                for j in range(args.n_discriminator):
                                    torch.save(discriminators[j].state_dict(), adv_model_path.format(experiment_type, j, args.LAMBDA, args.diff_LAMBDA))
                        else:
                            if best_adv_epoch + 5 <= k:
                                break
                    for j in range(args.n_discriminator):
                        discriminators[j].load_state_dict(torch.load(adv_model_path.format(experiment_type, j, args.LAMBDA, args.diff_LAMBDA)))

                after_train = time.time()
                train_time_list.append(after_train-before_train)
                train_epochs_list.append(i)


                model.load_state_dict(torch.load(main_model_path))

                '''get_leakage_data(model, training_generator, './inlp_input/verb_full_train_{}_{}_{}_{}_{}_{}.pickle'.format(args.experiment_type, args.lr, args.batch_size, args.loss_type, args.lambda_weight, args.seed), device, args)
                get_leakage_data(model, validation_generator, './inlp_input/verb_full_val_{}_{}_{}_{}_{}_{}.pickle'.format(args.experiment_type, args.lr, args.batch_size, args.loss_type, args.lambda_weight, args.seed), device, args)
                get_leakage_data(model, test_generator, './inlp_input/verb_full_test_{}_{}_{}_{}_{}_{}.pickle'.format(args.experiment_type, args.lr, args.batch_size, args.loss_type, args.lambda_weight, args.seed), device, args)

                train_leakage_data = load_leakage_data('./inlp_input/verb_full_train_{}_{}_{}_{}_{}_{}.pickle'.format(args.experiment_type, args.lr, args.batch_size, args.loss_type, args.lambda_weight, args.seed))
                val_leakage_data = load_leakage_data('./inlp_input/verb_full_val_{}_{}_{}_{}_{}_{}.pickle'.format(args.experiment_type, args.lr, args.batch_size, args.loss_type, args.lambda_weight, args.seed))
                test_leakage_data = load_leakage_data('./inlp_input/verb_full_test_{}_{}_{}_{}_{}_{}.pickle'.format(args.experiment_type, args.lr, args.batch_size, args.loss_type, args.lambda_weight, args.seed))'''
                
                train_leakage_data = get_leakage_data(model, training_generator, './output/verb_train.pickle', device, args)
                val_leakage_data = get_leakage_data(model, validation_generator, './output/verb_val.pickle', device, args)
                test_leakage_data = get_leakage_data(model, test_generator, './output/verb_test.pickle', device, args)

                #train_leakage_data = load_leakage_data('./output/verb_train.pickle')
                #val_leakage_data = load_leakage_data('./output/verb_val.pickle')
                #test_leakage_data = load_leakage_data('./output/verb_test.pickle')  
                # Evaluation
                test_loss, preds, labels, p_labels = eval_main(model, test_generator, criterion, device, args)
                preds = np.array(preds)
                labels = np.array(labels)
                p_labels = np.array(p_labels)

                '''pos_pos = []
                pos_neg = []
                neg_pos = []
                neg_neg = []
                for i in range(0, len(labels)):
                    if labels[i] == 1:
                        if p_labels[i] == 1:
                            pos_pos.append(i)
                        else:
                            pos_neg.append(i)
                    else:
                        if p_labels[i] == 1:
                            neg_pos.append(i)
                        else:
                            neg_neg.append(i)

                random.shuffle(pos_pos)
                random.shuffle(pos_neg)
                random.shuffle(neg_pos)
                random.shuffle(neg_neg)
                number = 200
                average_number = number // 4

                all_index = []
                all_index.extend(pos_pos[:average_number])
                all_index.extend(pos_neg[:average_number])
                all_index.extend(neg_pos[:average_number])
                all_index.extend(neg_neg[:average_number])
                selected_labels = [1]*average_number*2
                selected_labels.extend([0]*average_number*2)
                selected_p_labels = [1]*average_number
                selected_p_labels.extend([0]*average_number)
                selected_p_labels.extend([1]*average_number)
                biased_classifier                                

                biased_classifier                                
                biased_classifier                                
                biased_classifier                                

                biased_classifier                                
                biased_classifier                                
                biased_classifier                                
                biased_classifier                                
                with open('./analysis/selected_protected_labels.txt', 'w') as f:
                    for i in range(0, len(selected_p_labels)):
                        f.write(str(selected_p_labels[i])+'\n')'''

                accuracy = accuracy_score(labels, preds)

                with open('/data/scratch/projects/punim0478/ailis/MSCOCO/Balanced-Datasets-Are-Not-Enough-master/verb_classification/data/verb_full_test_df.pickle', 'rb') as f:
                    test_raw_data = pickle.load(f)
            
                counter = Counter(test_raw_data['verb'])

                for tem in counter:
                    counter[tem] = counter[tem]/float(len(test_raw_data))

                #print(counter)
                verb_counter = dict()
                for tem in counter:
                    verb_counter[id2verb[tem]] = counter[tem]

                rms_diff, weighted_rms_diff, tprs = tpr_multi(preds, labels, id2verb, i2g, p_labels, verb_counter)
                print('rms diff', rms_diff, 'weighted rms diff', weighted_rms_diff)
                logits_leakage = leakage_logits(train_leakage_data, val_leakage_data, test_leakage_data)
                hidden_leakage = leakage_hidden(train_leakage_data, val_leakage_data, test_leakage_data)
                accumulate_rms_diff.append(rms_diff)
                accumulate_weighted_rms_diff.append(weighted_rms_diff)
                accumulate_leakage_logits.append(logits_leakage[1])
                accumulate_leakage_hidden.append(hidden_leakage[1])
                difference, min_performance, macro_average, minority_performance = get_group_metrics(preds, labels, p_labels)
                accumulate_acc.append([args.lr, args.batch_size, tem_lambda, args.lambda_weight, 100*accuracy, logits_leakage[1], hidden_leakage[1], rms_diff, weighted_rms_diff, difference, min_performance, macro_average, minority_performance])
                
                #accumulate_acc.append([args.lr, args.batch_size, tem_lambda, args.lambda_weight, 100*accuracy, logits_leakage[1], hidden_leakage[1], rms_diff, weighted_rms_diff])
                #correlation_plot(tprs, prof2fem, 'world')
                output_file.write(str(args.lr)+'\t'+str(args.batch_size)+'\t'+str(args.lambda_weight)+'\t'+str(100*accuracy)+'\t'+str(logits_leakage[1])+'\t'+str(hidden_leakage[1])+'\n')
                output_file.flush()

                #test_representation = leakage_evaluation(model, -1, training_generator, validation_generator, test_generator, device)
                #leakage_evaluation(model, 0, training_generator, validation_generator, test_generator, device)

                '''representation = [test_representation[i] for i in all_index]
                with open(args.representation_file, 'w') as f:
                    for i in range(0, len(representation)):
                        f.write(' '.join([str(t) for t in representation[i]])+'\n')'''
                count_runs+=1
                print('hello world', count_runs, datetime.now())
                #break
                
    output_file.close()
    print('====================================================================================')
    for i in range(0, len(accumulate_acc)):
        #print('lr', accumulate_acc[i][0], 'batch size', accumulate_acc[i][1], 'lambda_ratio', accumulate_acc[i][2], 'actual lambda', accumulate_acc[i][3], 'accuracy', accumulate_acc[i][4], 'logits leakage', accumulate_acc[i][5], 'hidden leakage', accumulate_acc[i][6], accumulate_acc[i][7], accumulate_acc[i][8])    
        print('lr', accumulate_acc[i][0], 'batch size', accumulate_acc[i][1], 'lambda_ratio', accumulate_acc[i][2], 'actual lambda', accumulate_acc[i][3], 'accuracy', accumulate_acc[i][4], 'logits leakage', accumulate_acc[i][5], 'hidden leakage', accumulate_acc[i][6], accumulate_acc[i][7], accumulate_acc[i][8], accumulate_acc[i][9], accumulate_acc[i][10], accumulate_acc[i][11], accumulate_acc[i][12])    

    for i in range(0, len(train_time_list)):
        print(train_time_list[i],train_epochs_list[i])
    #print(train_time_list)
    #print(train_epochs_list)
    #print('time used', sum(train_time_list)/float(len(train_epochs_list)))