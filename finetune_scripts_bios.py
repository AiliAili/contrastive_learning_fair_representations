import os,argparse,time
import numpy as np
from datetime import datetime
import random
import torch
import torch.backends.cudnn as cudnn
import torch.optim 
import torch.utils.data
import torch.utils.data.distributed
import torch.nn.functional as F
import time
import torch.nn as nn

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

from dataloaders.bios_dataset import BiosDataset
from networks.deepmoji_sa import DeepMojiModel
from networks.discriminator import Discriminator
from dataloaders.scheduler import BalancedBatchSampler


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

from sklearn.linear_model import SGDClassifier, LogisticRegression


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
        tags = batch[1].long()
        p_tags = batch[2].long()

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
            protected_loss = contrastive_loss(features_2, p_tags)
            loss*=args.lambda_weight
            loss-=(1-args.lambda_weight)*protected_loss
        elif args.loss_type == 'scl1':
            con_loss = contrastive_loss_2(features_2, tags)
            loss=con_loss
            #print(loss.item())
        elif args.loss_type == 'scl2':
            protected_loss = contrastive_loss(features_2, p_tags)
            loss=-protected_loss
            '''loss = 0
            for i in range(0, 28):
                indices = torch.where(tags == i)[0].cpu().numpy()
                protected_loss = contrastive_loss(features_2[indices], p_tags[indices])                
                loss-=protected_loss'''
        elif args.loss_type == 'scl1-scl2':
            con_loss = contrastive_loss_2(features_2, tags)
            protected_loss = contrastive_loss(features_2, p_tags)
            '''indices_0 = torch.where(p_tags == 0)[0].cpu().numpy()
            indices_1 = torch.where(p_tags == 1)[0].cpu().numpy()
            number = min(len(indices_0), len(indices_1))
            actual_number = int(0.1*number)
            indices = indices_0[:actual_number]+indices_1[:actual_number]
            protected_loss = contrastive_loss(features_2[indices], p_tags[indices])'''
            #loss = con_loss-protected_loss
            #loss = con_loss


            loss = 0
            con_loss = 0
            protected_loss = 0
            for i in range(0, 28):
                indices_0 = torch.where(tags == i)[0].cpu().numpy()
                indices_1 = torch.where(tags == abs(27-i))[0].cpu().numpy()
                indices = list(indices_0)+list(indices_1)
                if len(indices) > 0:
                    con_loss+=contrastive_loss(features_2[indices], tags[indices])
                    protected_loss+=contrastive_loss(features_2[indices], p_tags[indices])

            '''for i in range(0, 211):
                indices = torch.where(tags == i)[0].cpu().numpy()
                #print(Counter(p_tags[indices].cpu().numpy()))
                if len(indices) > 0 and len(Counter(p_tags[indices].cpu().numpy()))==2:              
                    protected_loss = contrastive_loss(features_2[indices], p_tags[indices])                  
                    loss-=(1-args.lambda_weight)*protected_loss''' 

            loss = con_loss-protected_loss

        elif args.loss_type == 'ce+scl1-scl2':
            loss = criterion(predictions, tags)
            con_loss = contrastive_loss(features_2, tags)
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
def eval_main(model, iterator, criterion, contrastive_loss, contrastive_loss_2, device, args):
    
    epoch_loss = 0
    
    model.eval()
    
    preds = []
    labels = []
    private_labels = []

    for batch in iterator:
        
        text = batch[0]

        tags = batch[1]
        p_tags = batch[2]

        text = text.to(device)
        tags = tags.to(device).long()
        p_tags = p_tags.to(device).long()

        predictions, features_1, features_2, _ = model(text)
        if args.loss_type == 'ce':
            loss = criterion(predictions, tags) 
        else:
            loss = contrastive_loss_2(features_2, tags)-contrastive_loss(features_2, p_tags)
        
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
    #data_frame.to_pickle(filename)
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

    p2i, i2p = load_dictionary("./profession2index.txt")
    g2i, i2g = load_dictionary("./gender2index.txt")
    train_time_list = []
    train_epochs_list = []

    accumulate_acc = []
    accumulate_rms_diff = []
    accumulate_weighted_rms_diff = []
    accumulate_leakage_logits = []
    accumulate_leakage_hidden = []
    count_runs = 0
    accumulate_time = []

    selected_lambda = args.lambda_weight
    if True:
        if True:
            for tem_seed in [40, 41, 42, 43, 44, 45, 46, 47, 48, 49]:
            #if True:

                #tem_seed = 46
                args.seed = tem_seed
                tem_lambda = tem_seed
                print('============================================================')
                seed_everything(args.seed)

                print('bios batch size', args.batch_size, 'lr', args.lr, 'lambda_weight', args.lambda_weight, args.seed)
                #print('batch size', args.batch_size, 'lr', args.lr, 'lambda_weight', args.lambda_weight, args.lambda_1, args.lambda_2)
                # file names
                experiment_type = "adv_Diverse"
                
                # path to checkpoints
                main_model_path = "./official_models/bios_tem_finetune_model_{}_{}_{}_{}_{}_{}.pt".format(args.experiment_type, args.lr, args.batch_size, args.loss_type, args.lambda_weight, args.seed)

                # Device
                device = torch.device("cuda:"+str(args.device_id))
               
                # Load data
                balance_type = args.balance_type

                # Load data
                train_data = BiosDataset(args.data_path, "train", embedding_type = 'cls', balanced=False, balance_type=balance_type)
                dev_data = BiosDataset(args.data_path, "dev", embedding_type = 'cls',  balanced=False, balance_type=balance_type, shuffle=shuffle)
                test_data = BiosDataset(args.data_path, "test", embedding_type = 'cls', balanced=False, balance_type=balance_type, shuffle=shuffle)
                '''train_data = BiosDataset(args.data_path, "train", embedding_type = 'cls', )
                dev_data = BiosDataset(args.data_path, "dev", embedding_type = 'cls', )
                test_data = BiosDataset(args.data_path, "test", embedding_type = 'cls', )'''
                

                prof2fem = woman_profession_portion(train_data, dev_data, test_data, i2p, i2g)

                # Data loader
                training_generator = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=0)
                validation_generator = torch.utils.data.DataLoader(dev_data, batch_size=args.batch_size, shuffle=False, num_workers=0)
                test_generator = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=0)

                
                # Init model
                model = DeepMojiModel(args)
                #model = nn.DataParallel(model)
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
                                                                    contrastive_loss = contrastive_loss,
                                                                    contrastive_loss_2 = contrastive_loss_2,
                                                                    device = device, 
                                                                    args = args
                                                                    )
                best_loss = float('inf')
                best_acc = accuracy_score(valid_labels, valid_preds)
                best_epoch = 0

                start = time.time()

                for i in trange(args.num_epochs):
                    '''n_classes = 28
                    n_samples = 100
                    balanced_batch_sampler = BalancedBatchSampler(train_data, n_classes, n_samples)
                    training_generator = torch.utils.data.DataLoader(train_data, batch_sampler=balanced_batch_sampler)'''
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
                                                                        contrastive_loss = contrastive_loss,
                                                                        contrastive_loss_2 = contrastive_loss_2,
                                                                        device = device, 
                                                                        args = args
                                                                        )
                    valid_acc = accuracy_score(valid_labels, valid_preds)
                    # learning rate scheduler
                    scheduler.step(valid_loss)
                    #print('Valid loss', valid_loss, 'Valid acc', valid_acc, best_epoch, i, args.loss_type)

                    # early stopping
                    if valid_loss < best_loss:
                        best_acc = valid_acc
                        best_loss = valid_loss
                        best_epoch = i
                        torch.save(model.state_dict(), main_model_path)
                    else:
                        if best_epoch+5<=i:
                            break

                end = time.time()
                accumulate_time.append(end-start)

                model.load_state_dict(torch.load(main_model_path))

                '''get_leakage_data(model, training_generator, './inlp_input/verb_train_{}_{}_{}_{}_{}_{}.pickle'.format(args.experiment_type, args.lr, args.batch_size, args.loss_type, args.lambda_weight, args.seed), device, args)
                get_leakage_data(model, validation_generator, './inlp_input/verb_val_{}_{}_{}_{}_{}_{}.pickle'.format(args.experiment_type, args.lr, args.batch_size, args.loss_type, args.lambda_weight, args.seed), device, args)
                get_leakage_data(model, test_generator, './inlp_input/verb_test_{}_{}_{}_{}_{}_{}.pickle'.format(args.experiment_type, args.lr, args.batch_size, args.loss_type, args.lambda_weight, args.seed), device, args)

                train_leakage_data = load_leakage_data('./inlp_input/verb_train_{}_{}_{}_{}_{}_{}.pickle'.format(args.experiment_type, args.lr, args.batch_size, args.loss_type, args.lambda_weight, args.seed))
                val_leakage_data = load_leakage_data('./inlp_input/verb_val_{}_{}_{}_{}_{}_{}.pickle'.format(args.experiment_type, args.lr, args.batch_size, args.loss_type, args.lambda_weight, args.seed))
                test_leakage_data = load_leakage_data('./inlp_input/verb_test_{}_{}_{}_{}_{}_{}.pickle'.format(args.experiment_type, args.lr, args.batch_size, args.loss_type, args.lambda_weight, args.seed))'''


                train_leakage_data = get_leakage_data(model, training_generator, './output/bios_train.pickle', device, args)
                #test_leakage_data = get_leakage_data(model, test_generator, './inlp_input/verb_test_{}_{}_{}_{}_{}_{}_{}.pickle'.format(args.experiment_type, args.balance_type, args.lr, args.batch_size, args.loss_type, args.lambda_weight, args.seed), device, args)
                val_leakage_data = get_leakage_data(model, validation_generator, './output/bios_val.pickle', device, args)
                test_leakage_data = get_leakage_data(model, test_generator, './output/bios_test.pickle', device, args)

                start = time.time()

                train_hidden = train_leakage_data[1]
                train_labels = train_leakage_data[2]
                val_hidden = val_leakage_data[1]
                val_labels = val_leakage_data[2]
                p_val_labels = val_leakage_data[3]
                test_hidden = test_leakage_data[1]
                test_labels = test_leakage_data[2]
                p_test_labels = test_leakage_data[3]
                clf = LogisticRegression(warm_start = True, penalty = 'l2',
                        solver = "saga", multi_class = 'multinomial', fit_intercept = False,
                        verbose = False, n_jobs = 90, random_state = 1, max_iter = 10)#100)

                clf.fit(train_hidden[:], train_labels[:])
                accuracy = clf.score(test_hidden, test_labels)
                preds = clf.predict(test_hidden)

                train_logits = clf.predict_proba(train_hidden) 
                val_logits = clf.predict_proba(val_hidden)       
                test_logits = clf.predict_proba(test_hidden)
                print("time: {}".format((time.time() - start)/float(60)))

                '''index_0 = []
                index_1 = []         
                for i in range(0, len(test_labels)):
                    if i2p[test_labels[i]] == 'model':
                        if p_test_labels[i] == 0:
                            index_0.append(i)
                        else:
                            index_1.append(i)

                print('hello world', len(index_0), len(index_1)) 

                number = 200
                random.shuffle(index_0)
                random.shuffle(index_1)
                all_index = []
                all_index.extend(index_0[:number])
                all_index.extend(index_1[:number])
                                
                all_labels = []
                all_p_labels = []
          
                for i in range(0, len(all_index)):
                    all_labels.append(test_labels[all_index[i]])
                    all_p_labels.append(p_test_labels[all_index[i]])
                representation = [test_leakage_data[1][i] for i in all_index]
                                
                c = list(zip(all_index, all_labels, all_p_labels, representation))
                random.shuffle(c)
                all_index, selected_labels, selected_p_labels, representation = zip(*c)
                all_index = list(all_index)
                selected_labels = list(selected_labels)
                selected_p_labels = list(selected_p_labels)
                representation = list(representation)
                
                with open('./analysis/bios_model_selected_index.txt', 'w') as f:
                    for i in range(0, len(all_index)):
                        f.write(str(all_index[i])+'\n')

                with open('./analysis/bios_model_selected_labels.txt', 'w') as f:
                    for i in range(0, len(selected_labels)):
                        f.write(str(selected_labels[i])+'\n')
                
                with open('./analysis/bios_model_selected_protected_labels.txt', 'w') as f:
                    for i in range(0, len(selected_p_labels)):
                        f.write(str(selected_p_labels[i])+'\n')

                
                with open(args.representation_file, 'w') as f:
                    for i in range(0, len(representation)):
                        f.write(' '.join([str(t) for t in representation[i]])+'\n')'''


                with open('/data/scratch/projects/punim0478/ailis/nullspace_projection-master/data/biasbios/bios_test_df.pkl', 'rb') as f:
                #with open('/data/scratch/projects/punim0478/xudongh1/data/bios/bios_test_df.pkl', 'rb') as f:
                    test_raw_data = pickle.load(f)
            
                counter = Counter(test_raw_data['p'])
                for tem in counter:
                    counter[tem] = counter[tem]/float(len(test_raw_data))
        

                rms_diff, weighted_rms_diff, tprs = tpr_multi(preds, test_labels, i2p, i2g, p_test_labels, counter)
                print('accuracy', 100*accuracy, 'rms diff', rms_diff, 'weighted rms diff', weighted_rms_diff)

                logits_leakage = leakage_logits((train_logits, train_leakage_data[1], train_leakage_data[2], train_leakage_data[3]), (val_logits, val_leakage_data[1], val_leakage_data[2], val_leakage_data[3]),\
                (test_logits, test_leakage_data[1], test_leakage_data[2], test_leakage_data[3]))
                hidden_leakage = leakage_hidden(train_leakage_data, val_leakage_data, test_leakage_data)
                accumulate_rms_diff.append(rms_diff)
                accumulate_weighted_rms_diff.append(weighted_rms_diff)
                accumulate_leakage_logits.append(logits_leakage[1])
                accumulate_leakage_hidden.append(hidden_leakage[1])
                accumulate_acc.append([args.lr, args.batch_size, tem_lambda, args.lambda_weight, 100*accuracy, logits_leakage[1], hidden_leakage[1], rms_diff, weighted_rms_diff])
                #correlation_plot(tprs, prof2fem, 'world')
                
                count_runs+=1
                print('hello world', count_runs, datetime.now())
                print(accumulate_time[count_runs-1])
                break
                
    print('====================================================================================')
    for i in range(0, len(accumulate_acc)):
        print('lr', accumulate_acc[i][0], 'batch size', accumulate_acc[i][1], 'lambda_ratio', accumulate_acc[i][2], 'actual lambda', accumulate_acc[i][3], 'accuracy', accumulate_acc[i][4], 'logits leakage', accumulate_acc[i][5], 'hidden leakage', accumulate_acc[i][6], accumulate_acc[i][7], accumulate_acc[i][8])    

    print(accumulate_time)