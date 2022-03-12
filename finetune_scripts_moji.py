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

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

from dataloaders.deep_moji import DeepMojiDataset
from networks.deepmoji_sa import DeepMojiModel
from networks.discriminator import Discriminator


from tqdm import tqdm, trange
from networks.customized_loss import DiffLoss
from networks.contrastive_loss import Contrastive_Loss

from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from networks.eval_metrices_updated import leakage_evaluation, tpr_binary, leakage_hidden, leakage_logits

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

        text = text.to(device)
        tags = tags.to(device)
        p_tags = p_tags.to(device)
        
        optimizer.zero_grad()
        # main model predictions
        predictions, features_1, features_2, _ = model(text)

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
            con_loss = contrastive_loss(features_2, tags)
            loss=con_loss
        elif args.loss_type == 'scl2':
            protected_loss = contrastive_loss(features_2, p_tags)
            loss=-protected_loss
        elif args.loss_type == 'scl1-scl2':
            loss = criterion(predictions, tags)
            con_loss = contrastive_loss(features_2, tags)
            protected_loss = contrastive_loss(features_2, p_tags)
            loss = con_loss-protected_loss
        elif args.loss_type == 'ce+scl1-scl2':
            loss = criterion(predictions, tags)
            con_loss = contrastive_loss(features_2, tags)
            protected_loss = contrastive_loss(features_2, p_tags)
            loss*=args.lambda_weight
            loss+=(1-args.lambda_weight)*con_loss
            loss-=(1-args.lambda_weight)*protected_loss

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
        p_tags = p_tags.to(device).long()
        
        predictions, features_1, features_2, _ = model(text)

        loss = contrastive_loss(features_2, tags)-contrastive_loss(features_2, p_tags)
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
    data_frame['sentiment'] = labels
    data_frame['race'] = private_labels
    data_frame['second_last_representation'] = second_last_representation
    data_frame['predict'] = tem_preds
    data_frame.to_pickle(filename)
    accuracy = accuracy_score(labels, tem_preds)
    print('Potential', accuracy)

    X_logits = list(data_frame['prob'])
    X_hidden = list(data_frame['second_last_representation'])
    y = list(data_frame['sentiment'])
    gender_label = list(data_frame['race'])

    return (X_logits, X_hidden, y, gender_label)

def load_leakage_data(filename):
    data = pd.read_pickle(filename)
    X_logits = list(data['prob'])
    X_hidden = list(data['second_last_representation'])
    y = list(data['sentiment'])
    gender_label = list(data['race'])
        
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



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_fp16', action='store_true')
    parser.add_argument('--cuda', type=str)
    parser.add_argument('--hidden_size', type=int, default = 300)
    parser.add_argument('--emb_size', type=int, default = 2304)
    parser.add_argument('--num_classes', type=int, default = 2)
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
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--temperature', type=float, default=0.01)
    parser.add_argument('--batch_size', type=int, default = 512)    
    parser.add_argument('--seed', type=int, default = 46)
    parser.add_argument('--num_epochs', type=int, default = 60)
    parser.add_argument('--representation_file', default='./analysis/ce+scl1+scl2.txt', type=str, help='the file storing test representation before the classifier layer')
    parser.add_argument('--loss_type', default='ce', type=str, help='the type of loss we want to use')
    parser.add_argument('--lambda_weight', type=float, default=0.5, help='the weight for supervised contrastive learning over main labels')
    parser.add_argument('--experiment_type', default='adv_experiment_lambda', type=str, help='which types of experiments we are doing')
    parser.add_argument('--device_id', type=int, default = 0)
    args = parser.parse_args()

    train_time_list = []
    train_epochs_list = []

    accumulate_acc = []
    accumulate_rms_diff = []
    accumulate_weighted_rms_diff = []
    accumulate_leakage_logits = []
    accumulate_leakage_hidden = []
    count_runs = 0
    accumulate_time = []
    if True:
        if True:
            for tem_seed in [40, 41, 42, 43, 44, 45, 46, 47, 48, 49]:
            #for tem_seed in [46]:
           
                args.seed = tem_seed
                tem_lambda = tem_seed
                print('============================================================')
                seed_everything(args.seed)

                print('verb batch size', args.batch_size, 'lr', args.lr, 'lambda_weight', args.lambda_weight, args.seed)
                #print('batch size', args.batch_size, 'lr', args.lr, 'lambda_weight', args.lambda_weight, args.lambda_1, args.lambda_2)
                # file names
                experiment_type = "adv_Diverse"
                
                # path to checkpoints
                main_model_path = "./official_models/moji_tem_finetune_model_{}_{}_{}_{}_{}_{}.pt".format(args.experiment_type, args.lr, args.batch_size, args.loss_type, args.lambda_weight, args.seed)

                # Device
                device = torch.device("cuda:"+str(args.device_id))

                # Load data
                data_path = args.data_path
                train_data = DeepMojiDataset(args, data_path, "train", ratio=args.ratio, n = 100000)
                dev_data = DeepMojiDataset(args, data_path, "dev")
                test_data = DeepMojiDataset(args, data_path, "test")

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

                start = time.time()
                for i in trange(args.num_epochs):
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


                train_leakage_data = get_leakage_data(model, training_generator, './output/moji_train.pickle', device, args)
                #test_leakage_data = get_leakage_data(model, test_generator, './inlp_input/verb_test_{}_{}_{}_{}_{}_{}_{}.pickle'.format(args.experiment_type, args.balance_type, args.lr, args.batch_size, args.loss_type, args.lambda_weight, args.seed), device, args)
                val_leakage_data = get_leakage_data(model, validation_generator, './output/moji_val.pickle', device, args)
                test_leakage_data = get_leakage_data(model, test_generator, './output/moji_test.pickle', device, args)

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
                        verbose = False, n_jobs = 90, random_state = 1, max_iter = 10000
                        )

                clf.fit(train_hidden[:], train_labels[:])
                accuracy = clf.score(test_hidden, test_labels)
                preds = clf.predict(test_hidden)

                train_logits = clf.predict_proba(train_hidden) 
                val_logits = clf.predict_proba(val_hidden)       
                test_logits = clf.predict_proba(test_hidden)

                print("time: {}".format((time.time() - start)/float(60)))
                
                counter = Counter(test_data.y)
                for tem in counter:
                    counter[tem] = counter[tem]/float(len(test_data.y))

                rms_diff, weighted_rms_diff, tprs = tpr_binary(preds, test_labels, p_test_labels, counter)
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

                '''pos_pos = []
                pos_neg = []
                neg_pos = []
                neg_neg = []
                for i in range(0, len(test_labels)):
                    if test_labels[i] == 1:
                        if p_test_labels[i] == 1:
                            pos_pos.append(i)
                        else:
                            pos_neg.append(i)
                    else:
                        if p_test_labels[i] == 1:
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
                with open(args.representation_file, 'w') as f:
                    for i in range(0, len(representation)):
                        f.write(' '.join([str(t) for t in representation[i]])+'\n')'''
                
                count_runs+=1
                print('hello world', count_runs, datetime.now())
                print(accumulate_time[count_runs-1])
                #break

                
    print('====================================================================================')
    for i in range(0, len(accumulate_acc)):
        print('lr', accumulate_acc[i][0], 'batch size', accumulate_acc[i][1], 'lambda_ratio', accumulate_acc[i][2], 'actual lambda', accumulate_acc[i][3], 'accuracy', accumulate_acc[i][4], 'logits leakage', accumulate_acc[i][5], 'hidden leakage', accumulate_acc[i][6], accumulate_acc[i][7], accumulate_acc[i][8])    
    print(accumulate_time)