import os,argparse,time
import numpy as np
from datetime import datetime
import random
from random import shuffle
import torch
import torch.backends.cudnn as cudnn
import torch.optim 
import torch.utils.data
import torch.utils.data.distributed

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

#from dataloaders.bios_dataset import BiosDataset
from dataloaders.bios_dataset_binary import BiosDataset
from networks.deepmoji_sa import DeepMojiModel
#from networks.deepmoji_sa_xudong import DeepMojiModel
from networks.discriminator import Discriminator


from tqdm import tqdm, trange
from networks.customized_loss import DiffLoss
from networks.contrastive_loss import Contrastive_Loss

from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from networks.eval_metrices_updated import leakage_evaluation, tpr_multi, leakage_hidden, leakage_logits
from collections import defaultdict, Counter

from pathlib import Path, PureWindowsPath
from collections import defaultdict, Counter
import pickle
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr
import torch.nn.functional as F
import argparse
from sklearn.linear_model import SGDClassifier, LogisticRegression

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
        
        text = batch[0]
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

        text = text.to(device)
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
def train_epoch(model, discriminators, iterator, optimizer, criterion, contrastive_loss, contrastive_loss_2, device, args):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.train()
    for discriminator in discriminators:
        discriminator.train()

    # activate gradient reversal layer
    for discriminator in discriminators:
        discriminator.GR = True
    
    for batch in iterator:
        
        text = batch[0]
        #print('hello', text)
        tags = batch[1].long()
        #p_tags = batch[2].float()
        p_tags = batch[2].long()
        #print(p_tags)

        text = text.to(device)
        tags = tags.to(device)
        p_tags = p_tags.to(device)
        
        optimizer.zero_grad()
        # main model predictions
        predictions, features_1, features_2, _ = model(text)
        # main tasks loss
        loss = criterion(predictions, tags)
        

        if args.loss_type == 'ce':
            loss = criterion(predictions, tags) 
        elif args.loss_type == 'ce+scl1':
            loss = criterion(predictions, tags)
            con_loss = contrastive_loss(features_2, tags)
            loss*=args.lambda_weight
            loss+=(1-args.lambda_weight)*con_loss
        elif args.loss_type == 'ce-scl2':
            loss = criterion(predictions, tags)
            protected_loss = contrastive_loss(features_2, p_tags)#contrastive_loss(features_2, p_tags)
            loss*=args.lambda_weight
            loss-=(1-args.lambda_weight)*protected_loss
        elif args.loss_type == 'scl1':
            con_loss = contrastive_loss(features_2, tags)
            loss=(1-args.lambda_weight)*con_loss
        elif args.loss_type == 'scl2':
            protected_loss = contrastive_loss(features_2, p_tags)
            loss=-(1-args.lambda_weight)*protected_loss
        elif args.loss_type == 'scl1-scl2':
            loss = criterion(predictions, tags)
            con_loss = contrastive_loss(features_2, tags)
            protected_loss = contrastive_loss(features_2, p_tags)
            loss = 10*(1-args.lambda_weight)*con_loss-(1-args.lambda_weight)*protected_loss
        elif args.loss_type == 'ce+scl1-scl2':
            loss = criterion(predictions, tags)
            #con_loss = contrastive_loss(features_2, tags)
            con_loss = contrastive_loss_2(features_2, tags)
            protected_loss = contrastive_loss(features_2, p_tags)
            loss*=args.lambda_weight
            loss+=(1-args.lambda_weight)*con_loss
            loss-=(1-args.lambda_weight)*protected_loss
            #loss+=args.lambda_1*con_loss
            #loss-=args.lambda_2*protected_loss

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

        text = text.to(device)
        tags = tags.to(device).long()
        #p_tags = p_tags.to(device).float()
        p_tags = p_tags.to(device).long()

        predictions, features_1, features_2, _ = model(text)

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
    
    data_frame['prob'] = preds
    data_frame['profession_class'] = labels
    data_frame['gender_class'] = private_labels
    data_frame['second_last_representation'] = second_last_representation
    data_frame['predict'] = tem_preds
    data_frame.to_pickle(filename)
    accuracy = accuracy_score(labels, tem_preds)
    print('Potential', accuracy)

    X_logits = list(data_frame['prob'])
    X_hidden = list(data_frame['second_last_representation'])
    y = list(data_frame['profession_class'])
    gender_label = list(data_frame['gender_class'])

    return (X_logits, X_hidden, y, gender_label)

def load_leakage_data(filename):
    data = pd.read_pickle(filename)
    X_logits = list(data['prob'])
    X_hidden = list(data['second_last_representation'])
    y = list(data['profession_class'])
    gender_label = list(data['gender_class'])
        
    return (X_logits, X_hidden, y, gender_label)

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
        #print(profession, scores_dict)
        good_m, good_f = scores_dict["m"], scores_dict["f"]
        prof_total_f = prof_count_total[profession]["f"]
        prof_total_m = prof_count_total[profession]["m"]
        tpr_m = 100*(good_m) /float(prof_total_m)
        tpr_f = 100*(good_f) /float(prof_total_f)
        
        tprs[profession]["m"] = tpr_m
        tprs[profession]["f"] = tpr_f
        tprs_ratio.append(0)
        tprs_change[profession] = tpr_f - tpr_m
        #print(profession, (good_m+good_f)/float(prof_total_m+prof_total_f))
    
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
    plt.savefig("./tem1.png", dpi = 600)
    print("Correlation: {}; p-value: {}".format(*pearsonr(proportion_lst, tpr_lst)))
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_fp16', action='store_true')
    parser.add_argument('--cuda', type=str)
    parser.add_argument('--hidden_size', type=int, default = 300)
    parser.add_argument('--emb_size', type=int, default = 768)
    parser.add_argument('--num_classes', type=int, default = 28)
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
    parser.add_argument('--data_path', type=str, default='/data/scratch/projects/punim0478/xudongh1/data/bios/', help='directory containing the dataset')
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
    parser.add_argument('--experiment_type', default='adv_experiment_lambda', type=str, help='which types of experiments we are doing')
    parser.add_argument('--balance_type', default='g', type=str, help='which types of experiments we are doing')


    args = parser.parse_args()

    accumulate_accuracy = []
    accumulate_rms_diff = []
    accumulate_weighted_rms_diff = []
    accumulate_f1_macro = []
    accumulate_f1_micro = []
    accumulate_f1_weighted = []

    #p2i, i2p = load_dictionary("./profession2index.txt")
    g2i, i2g = load_dictionary("./gender2index.txt")

    i2p = {0: 'nurse', 1: 'surgeon'}
    #batch_list = [256, 512, 1024, 2048, 4096]
    #lr_list = [1e-5, 3e-5, 5e-5, 7e-5, 1e-4, 3e-4, 5e-4, 7e-4, 1e-3, 3e-3, 5e-3, 7e-3, 1e-2]
    #lambda_ratio_list = [1e-2, 5e-2, 1e-1, 5e-1, 1e0, 5e0, 1e1, 5e1, 1e2]
    #lambda_ratio_list = [1e-2, 5e-2, 1e-1, 5e-1, 1e0, 5e0, 1e1, 5e1, 1e2]
    #lambda_1_list = [1e-2, 5e-2, 1e-1, 5e-1, 1e0, 5e0]
    #lambda_2_list = [1e-2, 5e-2, 1e-1, 5e-1, 1e0, 5e0]
    batch_list = [256, 512, 1024, 2048]
    lr_list = [1e-5, 3e-5, 5e-5, 7e-5, 1e-4, 3e-4, 5e-4, 7e-4, 1e-3, 3e-3, 5e-3, 7e-3, 1e-2]
    #lr_list = [7e-5]
    lambda_ratio_list = [1e-2, 5e-2, 1e-1, 5e-1, 1e0, 5e0, 1e1, 5e1, 1e2]

    adv_batch_list = [256, 512, 1024, 2048]
    adv_lr_list = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0]
    adv_lambda_list = [1e-2, 5e-2, 1e-1, 5e-1, 1e0]
    adv_diff_lambda_list = [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4]

    accumulate_acc = []
    accumulate_rms_diff = []
    accumulate_weighted_rms_diff = []
    accumulate_leakage_logits = []
    accumulate_leakage_hidden = []
    count_runs = 0
    #output_file = open(args.representation_file, 'w')
    #for tem_lambda in [1e-5, 1e-4, 1e-3, 1e-2]:
    #for tem_lambda in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, valid_acc0.8, 0.9, 1.0]:
    #for batch_size in [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]:
    #for tem_seed in [40, 41, 42, 43, 44, 45, 46, 47, 48, 49]:
    #for tem_batch in batch_list:
    #for tem_1 in lambda_1_list:
    #    for tem_2 in lambda_2_list:

    #for tem_batch in adv_batch_list:
    #    for tem_lr in adv_lr_list:
    #        for tem_lambda in adv_lambda_list:
    #for tem_batch in batch_list:
    #    for tem_lr in lr_list:
    #        for tem_lambda in lambda_ratio_list:  
    selected_lambda = args.lambda_weight
    if True:
        if True:
            #for tem_lambda in adv_diff_lambda_list:
            #for tem_lambda in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            #for tem_lambda in [1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1e0]:
            #for tem_lambda in lambda_ratio_list:
            #for tem_temparature in [0.005, 0.007, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]:
            for tem_seed in [40, 41, 42, 43, 44, 45, 46, 47, 48, 49]:
            #for tem_lambda in adv_lambda_list:
            #if True:
                print('============================================================')
                #args.lambda_weight = 1/float(1+selected_lambda)
                #args.lr = tem_lr
                #args.batch_size = tem_batch

                '''args.LAMBDA = tem_lambda
                args.lambda_weight = tem_lambda
                args.lr = tem_lr
                args.batch_size = tem_batch'''
                #args.diff_LAMBDA = tem_lambda
                #tem_lambda = 1/float(1+selected_lambda)
                #args.lambda_weight = 1/float(1+args.lambda_weight)
                #args.LAMBDA = tem_lambda
                #args.diff_LAMBDA = tem_lambda
                args.lambda_weight = 1/float(1+selected_lambda)
                #args.lambda_weight = 1/float(1+tem_lambda)
                #args.lambda_weight = tem_lambda
                #args.LAMBDA = tem_lambda
                #args.temperature = tem_temparature
                #tem_lambda = tem_temparature
                tem_seed = 46
                tem_lambda = tem_seed
                args.seed = tem_seed
                print('============================================================')

                print('bios batch size', args.batch_size, 'lr', args.lr, 'lambda_weight', args.lambda_weight, args.seed)
                #print('batch size', args.batch_size, 'lr', args.lr, 'lambda_weight', args.lambda_weight, args.lambda_1, args.lambda_2)
                seed_everything(args.seed)
                #args.lambda_weight = 0.5

                # file names
                experiment_type = "adv_Diverse"
                
                # path to checkpoints
                main_model_path = "./trained_models/bios_model_{}_{}_{}_{}_{}_{}.pt".format(args.experiment_type, args.lr, args.batch_size, args.loss_type, args.lambda_weight, args.seed)
                #main_model_path = "./biographyFull_adv_model.pt"
                adv_model_path = "./official_models/bios_discriminator_{}_{}_{}_{}.pt"

                # Device
                device = torch.device("cuda:"+str(args.device_id))

                balance_type = args.balance_type

                # Load data
                train_data = BiosDataset(args.data_path, "train", embedding_type = 'cls', balanced=False, balance_type=balance_type)
                dev_data = BiosDataset(args.data_path, "dev", embedding_type = 'cls',  balanced=False, balance_type=balance_type, shuffle=shuffle)
                test_data = BiosDataset(args.data_path, "test", embedding_type = 'cls', balanced=False, balance_type=balance_type, shuffle=shuffle)


                prof2fem = woman_profession_portion(train_data, dev_data, test_data, i2p, i2g)

                # Data loader
                training_generator = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=0)
                validation_generator = torch.utils.data.DataLoader(dev_data, batch_size=args.batch_size, shuffle=False, num_workers=0)
                test_generator = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=0)

                # Init model
                model = DeepMojiModel(args)
                model = model.to(device)

                # Init discriminators
                # Number of discriminators
                n_discriminator = args.n_discriminator

                discriminators = [Discriminator(args, args.hidden_size, 2) for _ in range(n_discriminator)]
                discriminators = [dis.to(device) for dis in discriminators]

                diff_loss = DiffLoss()
                args.diff_loss = diff_loss

                contrastive_loss = Contrastive_Loss(device=device, temperature=args.temperature, base_temperature= args.temperature)
                contrastive_loss_2 = Contrastive_Loss(device=device, temperature=1*args.temperature, base_temperature= 1*args.temperature)

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

                for i in trange(args.num_epochs):
                    train_epoch(
                                model = model, 
                                discriminators = discriminators, 
                                iterator = training_generator, 
                                optimizer = optimizer, 
                                criterion = criterion, 
                                contrastive_loss = contrastive_loss,
                                contrastive_loss_2 = contrastive_loss_2,
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
                    # learning rate scheduler
                    scheduler.step(valid_loss)
                    #print('Valid loss', valid_loss, 'Valid acc', valid_acc, best_epoch, i, args.loss_type)

                    #test_loss, preds, labels, p_labels = eval_main(model, test_generator, criterion, device, args)
                    #accuracy = accuracy_score(labels, preds)
                    #print('overall accuracy', accuracy)

                    # early stopping
                    if valid_loss < best_loss:
                        best_acc = valid_acc
                        best_loss = valid_loss
                        best_epoch = i
                        torch.save(model.state_dict(), main_model_path)
                    else:
                        if best_epoch+30<=i:
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

                model.load_state_dict(torch.load(main_model_path))
                #print('load model')
                        
                '''get_leakage_data(model, training_generator, './inlp_input/bios_train_{}_{}_{}_{}_{}_{}.pickle'.format(args.experiment_type, args.lr, args.batch_size, args.loss_type, args.lambda_weight, args.seed), device, args)
                get_leakage_data(model, validation_generator, './inlp_input/bios_val_{}_{}_{}_{}_{}_{}.pickle'.format(args.experiment_type, args.lr, args.batch_size, args.loss_type, args.lambda_weight, args.seed), device, args)
                get_leakage_data(model, test_generator, './inlp_input/bios_test_{}_{}_{}_{}_{}_{}.pickle'.format(args.experiment_type, args.lr, args.batch_size, args.loss_type, args.lambda_weight, args.seed), device, args)

                train_leakage_data = load_leakage_data('./inlp_input/bios_train_{}_{}_{}_{}_{}_{}.pickle'.format(args.experiment_type, args.lr, args.batch_size, args.loss_type, args.lambda_weight, args.seed))
                val_leakage_data = load_leakage_data('./inlp_input/bios_val_{}_{}_{}_{}_{}_{}.pickle'.format(args.experiment_type, args.lr, args.batch_size, args.loss_type, args.lambda_weight, args.seed))
                test_leakage_data = load_leakage_data('./inlp_input/bios_test_{}_{}_{}_{}_{}_{}.pickle'.format(args.experiment_type, args.lr, args.batch_size, args.loss_type, args.lambda_weight, args.seed))'''

                train_leakage_data = get_leakage_data(model, training_generator, './output/bios_train.pickle', device, args)
                #val_leakage_data = get_leakage_data(model, validation_generator, './inlp_input/bios_val_{}_{}_{}_{}_{}_{}_{}.pickle'.format(args.experiment_type, args.balance_type, args.lr, args.batch_size, args.loss_type, args.lambda_weight, args.seed), device, args)
                #test_leakage_data = get_leakage_data(model, test_generator, './inlp_input/bios_test_{}_{}_{}_{}_{}_{}_{}.pickle'.format(args.experiment_type, args.balance_type, args.lr, args.batch_size, args.loss_type, args.lambda_weight, args.seed), device, args)
                val_leakage_data = get_leakage_data(model, validation_generator, './output/bios_val.pickle', device, args)
                test_leakage_data = get_leakage_data(model, test_generator, './output/bios_test.pickle', device, args)

                #train_leakage_data = get_leakage_data(model, training_generator, './inlp_input/bios_binary_train_{}_{}_{}_{}_{}_{}_{}.pickle'.format(args.experiment_type, args.balance_type, args.lr, args.batch_size, args.loss_type, args.lambda_weight, args.seed), device, args)
                #val_leakage_data = get_leakage_data(model, validation_generator, './inlp_input/bios_binary_val_{}_{}_{}_{}_{}_{}_{}.pickle'.format(args.experiment_type, args.balance_type, args.lr, args.batch_size, args.loss_type, args.lambda_weight, args.seed), device, args)
                #test_leakage_data = get_leakage_data(model, test_generator, './inlp_input/bios_binary_test_{}_{}_{}_{}_{}_{}_{}.pickle'.format(args.experiment_type, args.balance_type, args.lr, args.batch_size, args.loss_type, args.lambda_weight, args.seed), device, args)

                #train_leakage_data = load_leakage_data('./output/bios_adv_train.pickle')
                #val_leakage_data = load_leakage_data('./output/bios_adv_val.pickle')
                #test_leakage_data = load_leakage_data('./output/bios_adv_test.pickle')  
                '''train_hidden = train_leakage_data[1]
                train_labels = train_leakage_data[2]
                test_hidden = test_leakage_data[1]
                test_labels = test_leakage_data[2]
                clf = LogisticRegression(warm_start = True, penalty = 'l2',
                        solver = "saga", multi_class = 'multinomial', fit_intercept = False,
                        verbose = 5, n_jobs = 90, random_state = 1, max_iter = 40
                        )

                clf.fit(train_hidden[:], train_labels[:])
                print('hello world', clf.score(test_hidden, test_labels))'''

                # Evaluation
                test_loss, preds, labels, p_labels = eval_main(model, test_generator, criterion, device, args)
                preds = np.array(preds)
                labels = np.array(labels)
                p_labels = np.array(p_labels)      

                '''index_0 = []
                index_1 = []         
                for i in range(0, len(labels)):
                    if i2p[labels[i]] == 'nurse':
                        if p_labels[i] == 0:
                            index_0.append(i)
                        else:
                            index_1.append(i)

                print('hello world', len(index_0), len(index_1)) 

                number = 100
                random.shuffle(index_0)
                random.shuffle(index_1)
                all_index = []
                all_index.extend(index_0[:number])
                all_index.extend(index_1[:number])
                                
                all_labels = []
                all_p_labels = []
          
                for i in range(0, len(all_index)):
                    all_labels.append(labels[all_index[i]])
                    all_p_labels.append(p_labels[all_index[i]])
                representation = [test_leakage_data[1][i] for i in all_index]
                                
                c = list(zip(all_index, all_labels, all_p_labels, representation))
                random.shuffle(c)
                all_index, selected_labels, selected_p_labels, representation = zip(*c)
                all_index = list(all_index)
                selected_labels = list(selected_labels)
                selected_p_labels = list(selected_p_labels)
                representation = list(representation)
                with open('./analysis/bios_nurse_surgeon_selected_index.txt', 'w') as f:
                    for i in range(0, len(all_index)):
                        f.write(str(all_index[i])+'\n')

                with open('./analysis/bios_nurse_surgeon_selected_labels.txt', 'w') as f:
                    for i in range(0, len(selected_labels)):
                        f.write(str(selected_labels[i])+'\n')
                
                with open('./analysis/bios_nurse_surgeon_selected_protected_labels.txt', 'w') as f:
                    for i in range(0, len(selected_p_labels)):
                        f.write(str(selected_p_labels[i])+'\n')

                
                with open(args.representation_file, 'w') as f:
                    for i in range(0, len(representation)):
                        f.write(' '.join([str(t) for t in representation[i]])+'\n')'''

                pos_pos = []
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
                selected_p_labels.extend([0]*average_number)
                representation = [test_leakage_data[1][i] for i in all_index]

                c = list(zip(all_index, selected_labels, selected_p_labels, representation))
                random.shuffle(c)
                all_index, selected_labels, selected_p_labels, representation = zip(*c)
                all_index = list(all_index)
                selected_labels = list(selected_labels)
                selected_p_labels = list(selected_p_labels)
                representation = list(representation)
                with open('./analysis/bios_binary_nurse_surgeon_selected_index.txt', 'w') as f:
                    for i in range(0, len(all_index)):
                        f.write(str(all_index[i])+'\n')

                with open('./analysis/bios_binary_nurse_surgeon_selected_labels.txt', 'w') as f:
                    for i in range(0, len(selected_labels)):
                        f.write(str(selected_labels[i])+'\n')
                
                with open('./analysis/bios_binary_nurse_surgeon_selected_protected_labels.txt', 'w') as f:
                    for i in range(0, len(selected_p_labels)):
                        f.write(str(selected_p_labels[i])+'\n')

                
                with open(args.representation_file, 'w') as f:
                    for i in range(0, len(representation)):
                        f.write(' '.join([str(t) for t in representation[i]])+'\n')



                accuracy = accuracy_score(labels, preds)            

                '''with open('/data/scratch/projects/punim0478/ailis/nullspace_projection-master/data/biasbios/bios_test_df.pkl', 'rb') as f:
                    test_raw_data = pickle.load(f)
            
                counter = Counter(test_raw_data['p'])
                for tem in counter:
                    counter[tem] = counter[tem]/float(len(test_raw_data))'''

                test_data = BiosDataset(args.data_path, "test", embedding_type = 'cls', balanced=False, balance_type=balance_type, shuffle=shuffle)
                tem_counter = Counter(test_data.y)

                for tem in tem_counter:
                    tem_counter[tem] = tem_counter[tem]/float(len(test_data))

                counter = dict()
                for tem in tem_counter:
                    counter[i2p[tem]] = tem_counter[tem]

                #print(counter)

                rms_diff, weighted_rms_diff, tprs = tpr_multi(preds, labels, i2p, i2g, p_labels, counter)
                #for tem in tprs:
                #    print(tem, abs(tprs[tem]))

                print('rms diff', rms_diff, 'weighted rms diff', weighted_rms_diff)
                logits_leakage = leakage_logits(train_leakage_data, val_leakage_data, test_leakage_data)
                hidden_leakage = leakage_hidden(train_leakage_data, val_leakage_data, test_leakage_data)
                accumulate_rms_diff.append(rms_diff)
                accumulate_weighted_rms_diff.append(weighted_rms_diff)
                accumulate_leakage_logits.append(logits_leakage[1])
                accumulate_leakage_hidden.append(hidden_leakage[1])

                accumulate_acc.append([args.lr, args.batch_size, tem_lambda, args.lambda_weight, 100*accuracy, logits_leakage[1], hidden_leakage[1], rms_diff, weighted_rms_diff])
                #print(tprs, prof2fem)
                #correlation_plot(tprs, prof2fem, 'hello')
                #output_file.write(str(args.lr)+'\t'+str(args.batch_size)+'\t'+str(args.lambda_weight)+'\t'+str(100*accuracy)+'\t'+str(logits_leakage[1])+'\t'+str(hidden_leakage[1])+'\n')
                #output_file.flush()
            
                '''representation = [test_representation[i] for i in all_index]
                with open(args.representation_file, 'w') as f:
                    for i in range(0, len(representation)):
                        f.write(' '.join([str(t) for t in representation[i]])+'\n')'''

                count_runs+=1
                print('hello world', count_runs, datetime.now())
                break
                

    #output_file.close()
    print(balance_type)
    print('====================================================================================')
    for i in range(0, len(accumulate_acc)):
        print('lr', accumulate_acc[i][0], 'batch size', accumulate_acc[i][1], 'lambda_ratio', accumulate_acc[i][2], 'actual lambda', accumulate_acc[i][3], 'accuracy', accumulate_acc[i][4], 'logits leakage', accumulate_acc[i][5], 'hidden leakage', accumulate_acc[i][6], accumulate_acc[i][7], accumulate_acc[i][8])    
