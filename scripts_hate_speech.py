import os,argparse,time
import numpy as np
from datetime import datetime
import random
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

from dataloaders.hate_speech import HateSpeechDataset
from networks.deepmoji_sa import DeepMojiModel
from networks.discriminator import Discriminator


from tqdm import tqdm, trange
from networks.customized_loss import DiffLoss
from networks.contrastive_loss import Contrastive_Loss
from networks.center_loss import CenterLoss

from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
from networks.eval_metrices_updated import leakage_evaluation, tpr_binary, leakage_hidden, leakage_logits
from pathlib import Path, PureWindowsPath
from collections import defaultdict, Counter
import pandas as pd
import argparse
from random import shuffle
import time

monitor_micro_f1 = []
monitor_macro_f1 = []
monitor_weighted_f1 = []

monitor_class_distribution = []
monitor_per_class_f1 = []
monitor_group_0_percentage = []
monitor_group_1_percentage = []
monitor_per_class_group_0_f1 = []
monitor_per_class_group_1_f1 = []

# train a discriminator 1 epoch
def adv_train_epoch(model, discriminators, iterator, adv_optimizers, criterion, device, args):
    """"
    Train the discriminator to get a meaningful gradient
    """
    #print('start adv training')
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
def train_epoch(model, discriminators, iterator, optimizer, criterion,  contrastive_loss, contrastive_loss_2, center_loss, device, args):
    
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
        weights = batch[3]

        text = text.to(device)
        tags = tags.to(device)
        p_tags = p_tags.to(device)
        weights = weights.to(device)

        
        optimizer.zero_grad()
        # main model predictions
        predictions, features_1, features_2, features = model(text)
        # main tasks loss
        loss = criterion(predictions, tags)

        if args.mode == 'vanilla':
            loss = criterion(predictions, tags)
        elif args.mode == 'rw':
            loss = (criterion(predictions, tags)*weights).mean()
        elif args.mode == 'ds':
            loss = criterion(predictions, tags)
        elif args.mode == 'difference':
            loss = criterion(predictions, tags) 
            for i in range(0, 2):
                indices_0 = set(torch.where(tags == i)[0].cpu().numpy()).intersection(set(torch.where(p_tags == 0)[0].cpu().numpy()))
                indices_0 = list(indices_0)
                indices_1 = set(torch.where(tags == i)[0].cpu().numpy()).intersection(set(torch.where(p_tags == 1)[0].cpu().numpy()))
                indices_1 = list(indices_1)
                if len(indices_0) > 0 and len(indices_1) > 0:
                    tem_0 = criterion(predictions[indices_0], tags[indices_0])
                    tem_1 = criterion(predictions[indices_1], tags[indices_1])
                    if tem_0 > tem_1:
                        loss+=0.65*(tem_0-tem_1)
                    else:
                        loss+=0.65*(tem_1-tem_0)
            
        elif args.mode == 'mean':
            loss = criterion(predictions, tags)
            accu_loss = 0
            for i in range(0, args.num_classes):
                for j in range(0, 2):
                    indices = set(torch.where(tags == i)[0].cpu().numpy()).intersection(set(torch.where(p_tags == j)[0].cpu().numpy()))
                    indices = list(indices)
                    if len(indices) > 0:
                        loss_c_g = criterion(predictions[indices], tags[indices])
                        accu_loss+=0.65*abs(loss_c_g-loss)

            loss+=accu_loss
        else:
                
            if args.loss_type == 'ce':
                loss = criterion(predictions, tags)
                '''indices_0 = set(torch.where(p_tags == 0)[0].cpu().numpy())
                indices_0 = list(indices_0)
                indices_1 = set(torch.where(p_tags == 1)[0].cpu().numpy())
                indices_1 = list(indices_1)
                tem_0 = criterion(predictions[indices_0], tags[indices_0])
                tem_1 = criterion(predictions[indices_1], tags[indices_1])
                loss+= 0.6*abs(tem_0-tem_1)'''
                #print(loss)
                for i in range(0, 2):
                    indices_0 = set(torch.where(tags == i)[0].cpu().numpy()).intersection(set(torch.where(p_tags == 0)[0].cpu().numpy()))
                    indices_0 = list(indices_0)
                    indices_1 = set(torch.where(tags == i)[0].cpu().numpy()).intersection(set(torch.where(p_tags == 1)[0].cpu().numpy()))
                    indices_1 = list(indices_1)
                    indices = indices_0+indices_1
                    tem_loss = criterion(predictions[indices], tags[indices])
                    tem_0 = criterion(predictions[indices_0], tags[indices_0])
                    tem_1 = criterion(predictions[indices_1], tags[indices_1])
                    #loss+= 0.5*abs(tem_0-tem_1)
                    #loss+= 1.5*(tem_0-tem_1)*(tem_0-tem_1)
                    if tem_0 > tem_1:
                        loss+=0.65*(tem_0-tem_1)
                    else:
                        loss+=0.65*(tem_1-tem_0)
                    
                
                    #loss+=0.5*abs(tem_0-tem_loss)+0.5*abs(tem_1-tem_loss)

                    '''#loss+=2*criterion(predictions[indices_0], tags[indices_0])/criterion(predictions[indices_1], tags[indices_1])
                    #loss+=2*criterion(predictions[indices_1], tags[indices_1])/criterion(predictions[indices_0], tags[indices_0]) 
                    
                    similarity_0 = F.cosine_similarity(features[indices_0].unsqueeze(1), features[indices_0], dim=-1)
                    similarity_0 = similarity_0.view(-1)
                    similarity_loss_0 = similarity_0.sum()/float(similarity_0.shape[0])
                    
                    similarity_1 = F.cosine_similarity(features[indices_1].unsqueeze(1), features[indices_1], dim=-1)
                    similarity_1 = similarity_1.view(-1)
                    similarity_loss_1 = similarity_1.sum()/float(similarity_1.shape[0])

                    similarity_mix = F.cosine_similarity(features[indices_0].unsqueeze(1), features[indices_1], dim=-1)
                    similarity_mix = similarity_mix.view(-1)
                    similarity_loss_mix = similarity_mix.sum()/float(similarity_mix.shape[0])

                    loss+=1*similarity_loss_0
                    loss+=1*similarity_loss_1
                    loss-=1*similarity_loss_mix'''
                    '''center_0 = torch.sum(features_2[indices_0], dim=0)/float(len(indices_0))
                    center_1 = torch.sum(features_2[indices_1], dim=0)/float(len(indices_1))
                    #center_0 = features_2[indices_0[0]]
                    #center_1 = features_2[indices_1[0]]
                    center_0 = center_0.unsqueeze(0)
                    center_1 = center_1.unsqueeze(0)
                    
                    centers = [center_0, center_1]
                    centers = torch.cat((center_0, center_1), dim=0)
                    center_loss_tem = center_loss(features_2[indices], p_tags[indices], centers=centers)
                    #print(center_loss_tem)
                    #loss-=10*center_loss_tem'''
            elif args.loss_type == 'ce-mean':
                loss = criterion(predictions, tags)
                accu_loss = 0
                for i in range(0, args.num_classes):
                    for j in range(0, 2):
                        indices = set(torch.where(tags == i)[0].cpu().numpy()).intersection(set(torch.where(p_tags == j)[0].cpu().numpy()))
                        indices = list(indices)
                        loss_c_g = criterion(predictions[indices], tags[indices])
                        accu_loss+=0.65*abs(loss_c_g-loss)

                loss+=accu_loss

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
                '''for i in range(0, 2):
                    indices = torch.where(tags == i)[0].cpu().numpy()
                    protected_loss = contrastive_loss(features_2[indices], p_tags[indices])
                    loss-=(1-args.lambda_weight)*protected_loss'''
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
                loss = (1-args.lambda_weight)*con_loss-(1-args.lambda_weight)*protected_loss        
            elif args.loss_type == 'ce+scl1-scl2':
                loss = criterion(predictions, tags)
                loss*=args.lambda_weight
                con_loss = contrastive_loss(features_2, tags)
                loss+=(1-args.lambda_weight)*con_loss
                protected_loss = contrastive_loss(features_2, p_tags)
                loss-=(1-args.lambda_weight)*protected_loss
                '''#con_loss = contrastive_loss(F.normalize(predictions, dim=1), tags)
                con_loss = contrastive_loss(features_2, tags)
                loss+=(1-args.lambda_weight)*con_loss
                #protected_loss = contrastive_loss(F.normalize(predictions, dim=1), p_tags)
                protected_loss = contrastive_loss_2(features_2, p_tags)
                loss-=(1-args.lambda_weight)*protected_loss'''
                '''for i in range(0, 2):
                    indices = torch.where(tags == i)[0].cpu().numpy()
                    #print(type(indices))
                    #print(Counter(p_tags[indices].cpu().numpy()))
                    #p_index = torch.where((tags == i & p_tags==1))[0].cpu().numpy()
                    #n_index = torch.where((tags == i & p_tags==0))[0].cpu().numpy()
                    #p_index = (tags==i and p_tags==1)
                    index_all = torch.where(tags == i)[0].cpu().numpy()
                    
                    p_index = torch.where(p_tags == 1)[0].cpu().numpy()
                    #print(len(index_all), len(p_index))
                    p_index = np.intersect1d(index_all, p_index)
                    #print(p_index)
                    
                    index_all = torch.where(tags == i)[0].cpu().numpy()
                    n_index = torch.where(p_tags == 0)[0].cpu().numpy()
                    n_index = np.intersect1d(index_all, n_index)
                    #print(n_index)
                    number = min(len(p_index), len(n_index))
                    overall_index = np.array(list(p_index[:number])+list(n_index[:number]))
                    #print(overall_index)
                    indices = overall_index
                    
                    protected_loss = contrastive_loss(features_2[indices], p_tags[indices])
                    #protected_loss = contrastive_loss(F.normalize(predictions[indices], dim=1), p_tags[indices])
                    
                    loss-=(1-args.lambda_weight)*protected_loss'''
                '''man_index = torch.where(p_tags == 0)[0].cpu().numpy()
                woman_index = torch.where(p_tags == 1)[0].cpu().numpy()
                #print(len(man_index), len(woman_index))
                shuffle(man_index)
                shuffle(woman_index)
                selected_num = min(len(man_index), len(woman_index))
                new_features = torch.cat((features_2[man_index[:selected_num]], features_2[woman_index[:selected_num]]), 0)
                new_p_tags = torch.cat((p_tags[man_index[:selected_num]], p_tags[woman_index[:selected_num]]), 0)
                protected_loss = contrastive_loss(new_features, new_p_tags)
                loss-=(1-args.lambda_weight)*protected_loss'''
                '''for i in range(0, 2):
                    indices = torch.where(tags == i)[0].cpu().numpy()
                    
                    protected_loss = contrastive_loss(features_2[indices], p_tags[indices])
                    
                    loss-=(1-args.lambda_weight)*protected_loss'''

                '''con_loss = contrastive_loss(features_2, tags)
                protected_loss = contrastive_loss(features_2, p_tags)
                #con_loss = contrastive_loss(F.normalize(predictions, dim=1), tags)
                #protected_loss = contrastive_loss(F.normalize(predictions, dim=1), p_tags)
                loss*=args.lambda_weight
                loss+=(1-args.lambda_weight)*con_loss
                loss-=(1-args.lambda_weight)*protected_loss'''

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

        predictions, _, _, _ = model(text)
        
        loss = criterion(predictions, tags)
        if args.mode == 'rw':
            loss=loss.mean()
                        
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
    data_frame['age'] = private_labels
    data_frame['second_last_representation'] = second_last_representation
    data_frame['predict'] = tem_preds
    data_frame.to_pickle(filename)
    accuracy = accuracy_score(labels, tem_preds)
    print('Potential', accuracy)

    X_logits = list(data_frame['prob'])
    X_hidden = list(data_frame['second_last_representation'])
    y = list(data_frame['sentiment'])
    gender_label = list(data_frame['age'])

    return (X_logits, X_hidden, y, gender_label)

def load_leakage_data(filename):
    data = pd.read_pickle(filename)
    X_logits = list(data['prob'])
    X_hidden = list(data['second_last_representation'])
    y = list(data['sentiment'])
    gender_label = list(data['age'])
        
    return (X_logits, X_hidden, y, gender_label)

def get_group_metrics(preds, labels, p_labels, train_data):

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

    f1_group_0 = 100*f1_score(labels_0, preds_0, average=None)
    f1_group_1 = 100*f1_score(labels_1, preds_1, average=None)
    print('hello world', f1_group_0, f1_group_1)

    #micro
    micro_f1_0 = 100*f1_score(labels_0, preds_0, average='micro')
    micro_f1_1 = 100*f1_score(labels_1, preds_1, average='micro')
    monitor_micro_f1[-1].append(abs(micro_f1_0-micro_f1_1))
    monitor_micro_f1[-1].append(min(micro_f1_0, micro_f1_1))
    if len(preds_0) <= len(preds_1):
        monitor_micro_f1[-1].append(micro_f1_0)
    else:
        monitor_micro_f1[-1].append(micro_f1_1)

    #macro
    macro_f1_0 = 100*f1_score(labels_0, preds_0, average='macro')
    macro_f1_1 = 100*f1_score(labels_1, preds_1, average='macro')
    monitor_macro_f1[-1].append(abs(macro_f1_0-macro_f1_1))
    monitor_macro_f1[-1].append(min(macro_f1_0, macro_f1_1))
    if len(preds_0) <= len(preds_1):
        monitor_macro_f1[-1].append(macro_f1_0)
    else:
        monitor_macro_f1[-1].append(macro_f1_1)

    #weighted
    weighted_f1_0 = 100*f1_score(labels_0, preds_0, average='weighted')
    weighted_f1_1 = 100*f1_score(labels_1, preds_1, average='weighted')
    monitor_weighted_f1[-1].append(abs(weighted_f1_0-weighted_f1_1))
    monitor_weighted_f1[-1].append(min(weighted_f1_0, weighted_f1_1))
    if len(preds_0) <= len(preds_1):
        monitor_weighted_f1[-1].append(weighted_f1_0)
    else:
        monitor_weighted_f1[-1].append(weighted_f1_1)       


    if len(preds_0) <= len(preds_1):
        minority = accuracy_0
    else:
        minority = accuracy_1

    #print(len(preds_0), len(preds_1), accuracy_0, accuracy_1)

    #f1_per_class = 100*f1_score(labels, preds, average=None)
    matrix = confusion_matrix(labels, preds)
    f1_per_class = 100*matrix.diagonal()/matrix.sum(axis=1)
    #tem_class = []
    #class_proportion = []
    #tem_performance = []
    counter = Counter(train_data.y)
    for tem in counter:
        counter[tem] = counter[tem]/float(len(train_data))

    monitor_per_class_f1.append([])
    monitor_class_distribution.append([])
    for i in range(0, args.num_classes):
        monitor_per_class_f1[-1].extend([f1_per_class[i]])
        monitor_class_distribution[-1].append(100*counter[i])
        #tem_class.append(i)
        #class_proportion.append(counter[i]*100)
        #tem_performance.append(f1_per_class[i])

    '''print('================================')
    print(tem_class)
    print(tem_proportion)
    print(tem_performance)'''

    labels_0_train = []
    labels_1_train = []
    p_labels = train_data.private_label
    for i in range(0, len(p_labels)):
        if p_labels[i] == 0:
            labels_0_train.append(train_data.y[i])
        else:
            labels_1_train.append(train_data.y[i])

    #monitor_group_percentage.append([])
    #monitor_group_percentage[-1].extend([len(labels_0_train)/float(len(train_data)), len(labels_1_train)/float(len(train_data))])

    #f1_per_class_group_0 = 100*f1_score(labels_0, preds_0, average=None)
    matrix = confusion_matrix(labels_0, preds_0)
    f1_per_class_group_0 = 100*matrix.diagonal()/matrix.sum(axis=1)
    #tem_class = []
    #tem_proportion = []
    #tem_performance = []
    overall_counter = Counter(train_data.y)
    counter = Counter(labels_0_train)
    #print(counter)
    #distribution conditioned on y
    for tem in counter:
        counter[tem] = counter[tem]/float(overall_counter[tem])
    monitor_per_class_group_0_f1.append([])
    monitor_group_0_percentage.append([])
    for i in range(0, args.num_classes):
        monitor_per_class_group_0_f1[-1].extend([f1_per_class_group_0[i]])
        monitor_group_0_percentage[-1].append(100*counter[i])
        #tem_class.append(i)
        #tem_proportion.append(100*counter[i])
        #tem_performance.append(f1_per_class_group_0[i])
    
    '''print('================================')
    print(tem_class)
    print(tem_proportion)
    print(tem_performance)'''

    #f1_per_class_group_1 = 100*f1_score(labels_1, preds_1, average=None)
    matrix = confusion_matrix(labels_1, preds_1)
    f1_per_class_group_1 = 100*matrix.diagonal()/matrix.sum(axis=1)
    #tem_class = []
    #tem_proportion = []
    #tem_performance = []
    counter = Counter(labels_1_train)
    #print(counter)
    #distribution conditioned on y
    for tem in counter:
        counter[tem] = counter[tem]/float(overall_counter[tem])
    monitor_per_class_group_1_f1.append([])
    monitor_group_1_percentage.append([])
    for i in range(0, args.num_classes):
        monitor_per_class_group_1_f1[-1].extend([f1_per_class_group_1[i]])
        monitor_group_1_percentage[-1].append(100*counter[i])
        #tem_class.append(i)
        #tem_proportion.append(100*counter[i])
        #tem_performance.append(f1_per_class_group_1[i])

    '''print('================================')
    print(tem_class)
    print(tem_proportion)
    print(tem_performance)'''
    return abs(accuracy_0-accuracy_1), min(accuracy_0, accuracy_1), 0.5*(accuracy_0+accuracy_1), minority


def get_group_metrics_1(preds, labels, p_labels):

    f1_per_class = 100*f1_score(labels, preds, average=None)
    counter = Counter(labels)
    for tem in counter:
        counter[tem] = counter[tem]/float(len(labels))

    monitor_per_class_f1.append([])
    for i in range(0, args.num_classes):
        monitor_per_class_f1[-1].extend([counter[i], f1_per_class[i]])


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

    #micro
    micro_f1_0 = 100*f1_score(labels_0, preds_0, average='micro')
    micro_f1_1 = 100*f1_score(labels_1, preds_1, average='micro')
    monitor_micro_f1[-1].append(abs(micro_f1_0-micro_f1_1))
    monitor_micro_f1[-1].append(min(micro_f1_0, micro_f1_1))
    if len(preds_0) <= len(preds_1):
        monitor_micro_f1[-1].append(micro_f1_0)
    else:
        monitor_micro_f1[-1].append(micro_f1_1)


    #macro
    macro_f1_0 = 100*f1_score(labels_0, preds_0, average='macro')
    macro_f1_1 = 100*f1_score(labels_1, preds_1, average='macro')
    monitor_macro_f1[-1].append(abs(macro_f1_0-macro_f1_1))
    monitor_macro_f1[-1].append(min(macro_f1_0, macro_f1_1))
    if len(preds_0) <= len(preds_1):
        monitor_macro_f1[-1].append(macro_f1_0)
    else:
        monitor_macro_f1[-1].append(macro_f1_1)

    #weighted
    weighted_f1_0 = 100*f1_score(labels_0, preds_0, average='weighted')
    weighted_f1_1 = 100*f1_score(labels_1, preds_1, average='weighted')
    monitor_weighted_f1[-1].append(abs(weighted_f1_0-weighted_f1_1))
    monitor_weighted_f1[-1].append(min(weighted_f1_0, weighted_f1_1))
    if len(preds_0) <= len(preds_1):
        monitor_weighted_f1[-1].append(weighted_f1_0)
    else:
        monitor_weighted_f1[-1].append(weighted_f1_1)       

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_fp16', action='store_true')
    parser.add_argument('--cuda', type=str)
    parser.add_argument('--hidden_size', type=int, default = 300)
    parser.add_argument('--emb_size', type=int, default = 400)
    parser.add_argument('--num_classes', type=int, default = 2)
    parser.add_argument('--adv', action='store_true')
    parser.add_argument('--adv_level', type=int, default = -1)
    parser.add_argument('--lr', type=float, default=0.00003)
    parser.add_argument('--starting_power', type=int)
    parser.add_argument('--LAMBDA', type=float, default=0.7)
    parser.add_argument('--n_discriminator', type=int, default = 0)
    parser.add_argument('--adv_units', type=int, default = 256)
    parser.add_argument('--ratio', type=float, default=0.8)
    parser.add_argument('--DL', action='store_true')
    parser.add_argument('--diff_LAMBDA', type=float, default=1000)
    parser.add_argument('--data_path', type=str, default='/data/scratch/projects/punim0478/xudongh1/data/hate_speech/', help='directory containing the dataset')
    parser.add_argument('--temperature', type=float, default=0.01)
    parser.add_argument('--batch_size', type=int, default = 1024)    
    parser.add_argument('--seed', type=int, default = 46)
    parser.add_argument('--num_epochs', type=int, default = 60)
    parser.add_argument('--representation_file', default='./analysis/ce+scl1+scl2.txt', type=str, help='the file storing test representation before the classifier layer')
    parser.add_argument('--loss_type', default='ce', type=str, help='the type of loss we want to use')
    parser.add_argument('--lambda_weight', type=float, default=0.5, help='the weight for supervised contrastive learning over main labels')
    parser.add_argument('--experiment_type', default='adv_experiment_lambda', type=str, help='which types of experiments we are doing')
    parser.add_argument('--device_id', type=int, default = 0)
    parser.add_argument('--balance_type', default='stratified', type=str, help='which types of experiments we are doing')
    parser.add_argument('--mode', default='vanilla', type=str, help='which types of experiments we are doing')

    args = parser.parse_args()

    accumulate_rms_diff = []
    accumulate_weighted_rms_diff = []
    accumulate_leakage_logits = []
    accumulate_leakage_hidden = []
    

    batch_list = [256, 512, 1024, 2048, 4096]
    lr_list = [1e-5, 3e-5, 5e-5, 7e-5, 1e-4, 3e-4, 5e-4, 7e-4, 1e-3, 3e-3, 5e-3, 7e-3, 1e-2]
    lambda_ratio_list = [1e-2, 5e-2, 1e-1, 5e-1, 1e0, 5e0, 1e1, 5e1, 1e2]
    adv_batch_list = [256, 512, 1024, 2048]
    adv_lr_list = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0]
    adv_lambda_list = [1e-2, 5e-2, 1e-1, 5e-1, 1e0]
    adv_diff_lambda_list = [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4]
    #adv_diff_lambda_list = [1e3]

    #lambda_ratio_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0]
    accumulate_acc = []
    #for tem_lambda in [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4]:
    #for tem_seed in [40, 41, 42, 43, 44, 45, 46, 47, 48, 49]:
    #for tem_seed in [40]:
    #for tem_weight in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    #for tem_batch in batch_list:
    #    for tem_lr in lr_list:
    #if True:
    #    if True:
    #        if True:
    count_runs = 0
    accumulate_time = []
    #output_file = open(args.representation_file, 'w')
    '''for tem_batch in batch_list:
        for tem_lr in lr_list:
            for tem_lambda in lambda_ratio_list:'''
    #for tem_batch in adv_batch_list:
    #    for tem_lr in adv_lr_list:
    #        for tem_lambda in adv_lambda_list:
    selected_lambda = args.lambda_weight
    if True:
        if True:
            for tem_seed in [40, 41, 42, 43, 44, 45, 46, 47, 48, 49]:
            #for tem_lambda in adv_diff_lambda_list:
            #for tem_lambda in [1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1e0]:
            #for tem_lambda in lambda_ratio_list:
            #if True:
                print('============================================================')
                #args.diff_LAMBDA = tem_lambda
                #args.lambda_weight = tem_lambda

                #args.lambda_weight = 1/float(1+selected_lambda)
                #args.lambda_weight = tem_lambda
                #args.LAMBDA = tem_lambda
                #args.lambda_weight = tem_lambda
                #args.lr = tem_lr
                #args.batch_size = tem_batch

                #tem_seed = 46
                #tem_lambda = tem_seed
                
                args.seed = tem_seed
                #args.lambda_weight = 1/float(1+selected_lambda)
                tem_lambda = args.lambda_weight
                tem_lambda = tem_seed
                #args.LAMBDA = tem_lambda
                no_save = True
                print('hatespeech batch size', args.batch_size, 'lr', args.lr, 'lambda_weight', args.lambda_weight, args.seed)
                seed_everything(args.seed)
                #args.lambda_weight = tem_weight

                # file names
                experiment_type = "adv_Diverse"
                
                # path to checkpoints
                main_model_path = "./difference/hatespeech_model_{}_{}_{}_{}_{}_{}.pt".format(args.mode, args.lr, args.batch_size, args.loss_type, args.lambda_weight, args.seed)
                adv_model_path = "./official_models/hatespeech_discriminator_{}_{}_{}_{}.pt"
                
                author_private_label = "age"
                # Device
                device = torch.device("cuda:"+str(args.device_id))

                balance_type = args.balance_type

                if args.mode == 'ds':
                    balance_flag = True
                else:
                    balance_flag = False

                train_data = HateSpeechDataset(args, 
                                            args.data_path, 
                                            "train", 
                                            balanced = balance_flag,
                                            balance_type = balance_type,
                                            shuffle = shuffle,
                                            weight_scheme = 'joint',
                                            full_label_instances = True, 
                                            upsampling = False,
                                            private_label = author_private_label)
                        


                dev_data = HateSpeechDataset(args, 
                                        args.data_path, 
                                        "valid", 
                                        balanced = False,
                                        balance_type = balance_type,
                                        shuffle = shuffle,
                                        full_label_instances = True, 
                                        upsampling = False,
                                        private_label = author_private_label)

                test_data = HateSpeechDataset(args, 
                                        args.data_path, 
                                        "test", 
                                        balanced = False,
                                        balance_type = balance_type,
                                        shuffle = shuffle,
                                        full_label_instances = True, 
                                        upsampling = False,
                                        private_label = author_private_label)
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
                #center_loss = CenterLoss(num_classes=2, feat_dim=args.hidden_size, use_gpu=True)
                center_loss = contrastive_loss

                # Init optimizers
                LEARNING_RATE = args.lr
                optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)

                adv_optimizers = [Adam(filter(lambda p: p.requires_grad, dis.parameters()), lr=1e-1*LEARNING_RATE) for dis in discriminators]

                # Init learing rate scheduler
                scheduler = ReduceLROnPlateau(optimizer, mode = 'max', factor = 0.5, patience = 2)

                # Init criterion
                if args.mode == 'rw':
                    criterion = torch.nn.CrossEntropyLoss(reduction='none')

                else:
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
                    #args.loss_type = 'ce+scl1-scl2'
                    #args.lambda_weight = 1-i/float(60)
                    #print('hello', args.lambda_weight)
                    train_epoch(
                                model = model, 
                                discriminators = discriminators, 
                                iterator = training_generator, 
                                optimizer = optimizer, 
                                criterion = criterion, 
                                contrastive_loss = contrastive_loss,
                                contrastive_loss_2 = contrastive_loss_2,
                                center_loss = center_loss, 
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
                    # early stopping
                    if valid_loss < best_loss:
                        best_acc = valid_acc
                        best_loss = valid_loss
                        best_epoch = i
                        torch.save(model.state_dict(), main_model_path)
                    else:
                        if best_epoch+5<=i:
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
                    best_adv_loss = float('inf')
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
                            
                        #print('hellll', adv_valid_loss, best_adv_loss)
                        if adv_valid_loss < best_adv_loss or no_save == True:
                                best_adv_loss = adv_valid_loss
                                best_adv_epoch = k
                                no_save = False
                                for j in range(args.n_discriminator):
                                    torch.save(discriminators[j].state_dict(), adv_model_path.format(experiment_type, j, args.LAMBDA, args.diff_LAMBDA))
                        else:
                            if best_adv_epoch + 5 <= k:
                                break
                    for j in range(args.n_discriminator):
                        discriminators[j].load_state_dict(torch.load(adv_model_path.format(experiment_type, j, args.LAMBDA, args.diff_LAMBDA)))
                
                end = time.time()
                accumulate_time.append(end-start)

                model.load_state_dict(torch.load(main_model_path))
                            
                '''get_leakage_data(model, training_generator, './inlp_input/hatespeech_train_{}_{}_{}_{}_{}_{}.pickle'.format(args.experiment_type, args.lr, args.batch_size, args.loss_type, args.lambda_weight, args.seed), device, args)
                get_leakage_data(model, validation_generator, './inlp_input/hatespeech_val_{}_{}_{}_{}_{}_{}.pickle'.format(args.experiment_type, args.lr, args.batch_size, args.loss_type, args.lambda_weight, args.seed), device, args)
                get_leakage_data(model, test_generator, './inlp_input/hatespeech_test_{}_{}_{}_{}_{}_{}.pickle'.format(args.experiment_type, args.lr, args.batch_size, args.loss_type, args.lambda_weight, args.seed), device, args)

                train_leakage_data = load_leakage_data('./inlp_input/hatespeech_train_{}_{}_{}_{}_{}_{}.pickle'.format(args.experiment_type, args.lr, args.batch_size, args.loss_type, args.lambda_weight, args.seed))
                val_leakage_data = load_leakage_data('./inlp_input/hatespeech_val_{}_{}_{}_{}_{}_{}.pickle'.format(args.experiment_type, args.lr, args.batch_size, args.loss_type, args.lambda_weight, args.seed))
                test_leakage_data = load_leakage_data('./inlp_input/hatespeech_test_{}_{}_{}_{}_{}_{}.pickle'.format(args.experiment_type, args.lr, args.batch_size, args.loss_type, args.lambda_weight, args.seed))'''

                train_leakage_data = get_leakage_data(model, training_generator, './output/hatespeech_train.pickle', device, args)
                val_leakage_data = get_leakage_data(model, validation_generator, './output/hatespeech_val.pickle', device, args)
                test_leakage_data = get_leakage_data(model, test_generator, './output/hatespeech_test.pickle', device, args)
                #val_leakage_data = get_leakage_data(model, validation_generator, './inlp_input/hatespeech_val_{}_{}_{}_{}_{}_{}_{}.pickle'.format(args.experiment_type, args.balance_type, args.lr, args.batch_size, args.loss_type, args.lambda_weight, args.seed), device, args)
                #test_leakage_data = get_leakage_data(model, test_generator, './inlp_input/hatespeech_test_{}_{}_{}_{}_{}_{}_{}.pickle'.format(args.experiment_type, args.balance_type, args.lr, args.batch_size, args.loss_type, args.lambda_weight, args.seed), device, args)

                #train_leakage_data = load_leakage_data('./output/hatespeech_train.pickle')
                #val_leakage_data = load_leakage_data('./output/hatespeech_val.pickle')
               # test_leakage_data = load_leakage_data('./output/hatespeech_test.pickle')  
                
                # Evaluation
                test_loss, preds, labels, p_labels = eval_main(model, test_generator, criterion, device, args)
                preds = np.array(preds)
                labels = np.array(labels)
                p_labels = np.array(p_labels)
            
                counter = Counter(test_data.y)
                for tem in counter:
                    counter[tem] = counter[tem]/float(len(test_data.y))
                #print(counter)
                rms_diff, weighted_rms_diff, tprs = tpr_binary(preds, labels, p_labels, counter)
                print('rms diff', rms_diff, 'weighted rms diff', weighted_rms_diff)
                logits_leakage = leakage_logits(train_leakage_data, val_leakage_data, test_leakage_data)
                hidden_leakage = leakage_hidden(train_leakage_data, val_leakage_data, test_leakage_data)
                print('logits leakage', logits_leakage, 'hidden leakage', hidden_leakage)
                accumulate_rms_diff.append(rms_diff)
                accumulate_weighted_rms_diff.append(weighted_rms_diff)
                accumulate_leakage_logits.append(logits_leakage[1])
                accumulate_leakage_hidden.append(hidden_leakage[1])


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
                selected_p_labels.extend([0]*average_number)
                representation = [test_leakage_data[1][i] for i in all_index]
                selected_predictions = preds[all_index]

                c = list(zip(all_index, selected_predictions, selected_labels, selected_p_labels, representation))
                random.shuffle(c)
                all_index, selected_predictions, selected_labels, selected_p_labels, representation = zip(*c)
                all_index = list(all_index)
                selected_predictions = list(selected_predictions)
                selected_labels = list(selected_labels)
                selected_p_labels = list(selected_p_labels)
                representation = list(representation)
                with open('./analysis/hatespeech_selected_index.txt', 'w') as f:
                    for i in range(0, len(all_index)):
                        f.write(str(all_index[i])+'\n')

                with open('./analysis/hatespeech_selected_labels.txt', 'w') as f:
                    for i in range(0, len(selected_labels)):
                        f.write(str(selected_labels[i])+'\n')
                
                with open('./analysis/hatespeech_selected_protected_labels.txt', 'w') as f:
                    for i in range(0, len(selected_p_labels)):
                        f.write(str(selected_p_labels[i])+'\n')

                
                with open(args.representation_file, 'w') as f:
                    for i in range(0, len(representation)):
                        f.write(' '.join([str(t) for t in representation[i]])+'\n')

                print('=====================================================================================')
                rms_diff_tem, weighted_rms_diff_tem, tprs_tem = tpr_binary(selected_predictions, selected_labels, selected_p_labels, counter)
                print('rms diff', rms_diff_tem, 'weighted rms diff', weighted_rms_diff_tem)'''
                
                accuracy = accuracy_score(labels, preds)

                micro_f1 = 100*f1_score(labels, preds, average='micro')
                monitor_micro_f1.append([micro_f1])
                macro_f1 = 100*f1_score(labels, preds, average='macro')
                monitor_macro_f1.append([macro_f1])
                weighted_f1 = 100*f1_score(labels, preds, average='weighted')
                monitor_weighted_f1.append([weighted_f1])

                #accumulate_acc.append([args.lr, args.batch_size, 100*accuracy])
                difference, min_performance, macro_average, minority_performance = get_group_metrics(preds, labels, p_labels, train_data)
                accumulate_acc.append([args.lr, args.batch_size, tem_lambda, args.lambda_weight, 100*accuracy, logits_leakage[1], hidden_leakage[1], rms_diff, weighted_rms_diff, difference, min_performance, macro_average, minority_performance])
                #output_file.write(str(args.lr)+'\t'+str(args.batch_size)+'\t'+str(args.lambda_weight)+'\t'+str(100*accuracy)+'\t'+str(logits_leakage[1])+'\t'+str(hidden_leakage[1])+'\n')
                #output_file.flush()
                #test_representation, dev_leakage_hidden, test_leakage_hidden = leakage_evaluation(model, -1, training_generator, validation_generator, test_generator, device)
                #_, dev_leakage_output, test_leakage_output = leakage_evaluation(model, 0, training_generator, validation_generator, test_generator, device)
                count_runs+=1
                print('hello world', count_runs, datetime.now())
                print(accumulate_time[count_runs-1])
                '''representation = [test_representation[i] for i in all_index]
                with open(args.representation_file, 'w') as f:
                    for i in range(0, len(representation)):
                        f.write(' '.join([str(t) for t in representation[i]])+'\n')'''
                #break

    #output_file.close()
    print('====================================================================================')
    for i in range(0, len(accumulate_acc)):
        #print('lr', accumulate_acc[i][0], 'batch size', accumulate_acc[i][1], 'accuracy', accumulate_acc[i][2])
        print('lr', accumulate_acc[i][0], 'batch size', accumulate_acc[i][1], 'lambda_ratio', accumulate_acc[i][2], 'actual lambda', accumulate_acc[i][3], 'accuracy', accumulate_acc[i][4], 'logits leakage', accumulate_acc[i][5], 'hidden leakage', accumulate_acc[i][6], accumulate_acc[i][7], accumulate_acc[i][8], accumulate_acc[i][9], accumulate_acc[i][10], accumulate_acc[i][11], accumulate_acc[i][12])    

    print('micro')
    for i in range(0, len(monitor_micro_f1)):
        print(monitor_micro_f1[i][0], monitor_micro_f1[i][1], monitor_micro_f1[i][2], monitor_micro_f1[i][3])

    print('macro')
    for i in range(0, len(monitor_macro_f1)):
        print(monitor_macro_f1[i][0], monitor_macro_f1[i][1], monitor_macro_f1[i][2], monitor_macro_f1[i][3])

    print('weighted')
    for i in range(0, len(monitor_weighted_f1)):
        print(monitor_weighted_f1[i][0], monitor_weighted_f1[i][1], monitor_weighted_f1[i][2], monitor_weighted_f1[i][3])
        
    print('per class distribution')
    print(monitor_class_distribution[0])

    print('per class')
    for i in range(0, len(monitor_per_class_f1)):
        print(monitor_per_class_f1[i])

    print('group 0 distribution')
    print(monitor_group_0_percentage[0])

    print('group 0 per class')
    for i in range(0, len(monitor_per_class_group_0_f1)):
        print(monitor_per_class_group_0_f1[i])

    print('group 1 distribution')
    print(monitor_group_1_percentage[0])

    print('group 1 per class')
    for i in range(0, len(monitor_per_class_group_1_f1)):
        print(monitor_per_class_group_1_f1[i])

    #print(accumulate_time)