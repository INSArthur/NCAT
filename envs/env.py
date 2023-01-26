#!/usr/bin/python
# encoding: utf-8
import sys
import numpy as np
import pandas as pd
import sklearn.preprocessing as skPrep

from collections import OrderedDict
from util import *
import math
from collections import Counter
import copy as cp
import json
import os
import yaml
import torch
from torch.utils.data import Dataset, DataLoader
from EduCDM import EMIRT
from EduCDM import NCDM
from sklearn.metrics import roc_auc_score

# TODO:
class env(object):
    def __init__(self, args):
        self.T = args.T # ex : T=20
        self.data_name = args.data_name
        self.CDM = args.CDM
        self.rates = {}
        self.users = {}
        self.utypes = {}
        self.args = args
        self.device = torch.device("cpu")

        self.rates, self._item_num, self.know_map = self.load_data(os.path.join(self.args.data_path, self.args.data_name, self.args.data_name+".csv"))

        self.setup_train_test()
        self.sup_rates, self.query_rates = self.split_data(ratio=0.5) # sup_rates = items déjà posé, query_rates item à poser ?
        self.model, self.dataset = self.load_CDM()
        print(self.model)

    # Divise chaque utilisateur (que ce soit pour le training, validation ou evaluation set) en query set et support set
    def split_data(self, ratio=0.5):
        sup_rates, query_rates = {}, {}
        for u in self.rates:
            all_items = list(self.rates[u].keys())
            np.random.shuffle(all_items)
            sup_rates[u] = {it: self.rates[u][it] for it in all_items[:int(ratio*len(all_items))]}
            query_rates[u] = {it: self.rates[u][it] for it in all_items[int(ratio*len(all_items)):]}
        return sup_rates, query_rates

    def re_split_data(self, ratio=0.5):
        self.sup_rates, self.query_rates = self.split_data(ratio)

    @property
    def candidate_items(self):
        return set(self.sup_rates[self.state[0][0]].keys())

    @property
    def user_num(self):
        return len(self.rates) + 1 #TODO : check if +1 works

    @property
    def item_num(self):
        return self._item_num + 1

    @property
    def utype_num(self):
        return len(self.utypes) + 1

    def load_CDM(self):

        name = self.CDM
        # TODO : déclarer correctement tous les attributs de self qui sont créés dans leur fonctions respectives. Les faires passer dans les retours de fonction
        self.training_data = self.categorized_compacted_data[self.categorized_compacted_data['studentId'].isin(self.training)]

        # construction d'une matrice des scores du training set
        R = -1 * np.ones(shape=(self.training.shape[0], self._item_num))

        # Mapping des studentId du training set sur l'intervalle du training set :
        encoder_training_dataset = skPrep.OrdinalEncoder()
        categorized_training_dataset = encoder_training_dataset.fit_transform(self.training_data[['studentId', 'skill', 'problemId','correct']])
        categorized_training_dataset = np.array(categorized_training_dataset, dtype=int)
        R[categorized_training_dataset[:,0], categorized_training_dataset[:,2]] = categorized_training_dataset[:,3]


        if 'NCDM' in name:
            #TODO : implement this part
            model = 7
        elif 'EMIRT' in name:
            model = EMIRT(R, self.training.shape[0], self._item_num, dim=1, skip_value=-1)

            # name = self.CDM
        # CONFIG = yaml.load(open('./envs/pre_train/config.yml', 'r'), Loader=yaml.Loader)
        # CONFIG_DATA = yaml.load(open('./data/{}/info_filtered.yml'.format(self.data_name), 'r'), Loader=yaml.Loader)
        # cat_data = train_dataset(None, self.args.data_name, CONFIG_DATA['kc_maxid']+1, 'train', self.know_map)
        # best_model_path = './envs/model_file/{}/{}'.format(self.data_name, CONFIG[name]['best_model_path'])
        #
        # if 'NCD' in name:
        #     info = NCD_Info(name, CONFIG[name]['layers_fc_dim'], CONFIG[name]['layers_fc_dropout'])
        #     model = NCD(info, CONFIG_DATA['stu_maxid']+1, CONFIG_DATA['exer_maxid']+1, CONFIG_DATA['kc_maxid']+1, best_model_path).to(self.device)
        #
        # elif 'NCDM' in name:
        #     model =
        #
        # elif 'MIRT' in name:
        #     self.ismirt = True
        #     info = MIRT_Info(name, CONFIG[name]['dim'], CONFIG[name]['guess'])
        #     model = MIRT(info, CONFIG_DATA['stu_maxid']+1, CONFIG_DATA['exer_maxid']+1, CONFIG_DATA['kc_maxid']+1, best_model_path).to(self.device)
        #
        # elif 'IRT' in name and 'MIRT' not in name:
        #     info = IRT_Info(name, CONFIG[name]['guess'])
        #     model = IRT(info, CONFIG_DATA['stu_maxid']+1, CONFIG_DATA['exer_maxid']+1, CONFIG_DATA['kc_maxid']+1, best_model_path).to(self.device)

        return model, categorized_training_dataset

    # Split the users id into training, validation and evaluation  : [0-0.8*user_num:0.8*user_num-0.9*user_num:0.9*user_num-user_num]
    def setup_train_test(self):
        users = list(range(1, self.user_num)) #TODO : find origin user_num
        np.random.shuffle(users)
        self.training, self.validation, self.evaluation = np.split(np.asarray(users), [int(.8 * self.user_num - 1),
                                                                                       int(.9 * self.user_num - 1)])
    def load_data(self, path):
        data = pd.read_csv(path,low_memory=False)

        encoder_dataset = skPrep.OrdinalEncoder()
        encoder_dataset.fit(data[['studentId', 'skill', 'problemId']])
        categorized_data = encoder_dataset.transform(data[['studentId', 'skill', 'problemId']])

        data['studentId'] = categorized_data[:, 0]
        data['skill'] = categorized_data[:, 1]
        data['problemId'] = categorized_data[:, 2]

        know_map = {}

        s = data['studentId'].value_counts()
        s2 = s[s > 2 * self.T]
        self.categorized_compacted_data = data[data['studentId'].isin(s2.index)]

        max_itemid = int(self.categorized_compacted_data['problemId'].max())

        rates = dict.fromkeys(np.arange(self.categorized_compacted_data['studentId'].max()+1,dtype=int),{})


        for index, row in self.categorized_compacted_data.iterrows():
            rates[int(row['studentId'])][int(row['problemId'])] = int(row['correct'])
            know_map[int(row['problemId'])] = int(row['skill'])



        # with open(path, encoding='utf8') as i_f:
        #     stus = json.load(i_f) # list
        # rates = {}
        # items = set()
        # user_cnt = 0 # user count : donne 1 id unique à chaque utilisateur
        # know_map = {}
        #
        # for stu in stus:
        #     if stu['log_num'] < self.T * 2: # Si l'étudiant à répondu à moins de 2*T réponses, on ne l'enregistre pas
        #         continue
        #     user_cnt += 1
        #     rates[user_cnt] = {}
        #     for log in stu['logs']:
        #         rates[user_cnt][int(log['exer_id'])] = int(log['score'])
        #         items.add(int(log['exer_id']))
        #         know_map[int(log['exer_id'])] = log['knowledge_code']
        #

        return rates, max_itemid, know_map

    def reset(self):
        self.reset_with_users(np.random.choice(self.training))

    def reset_with_users(self, uid):
        self.state = [(uid,1), []]
        self.short = {}
        return self.state

    def step(self, action):
        assert action in self.sup_rates[self.state[0][0]] and action not in self.short
        reward, ACC, AUC, rate = self.reward(action)

        if len(self.state[1]) < self.T - 1:
            done = False
        else:
            done = True

        self.short[action] = 1
        t = self.state[1] + [[action, reward, done]]
        info = {"ACC": ACC,
                "AUC": AUC,
                "rate":rate}
        self.state[1].append([action, reward, done, info])
        return self.state, reward, done, info

    def reward(self, action):
        self.dataset.clear()
        items = [state[0] for state in self.state[1]] + [action]
        correct = [self.rates[self.state[0][0]][it] for it in items]
        self.dataset.add_record([self.state[0][0]]*len(items), items, correct)
        self.model.update(self.dataset, self.args.learning_rate, epoch=1)

        item_query = list(self.query_rates[self.state[0][0]].keys())
        correct_query = [self.rates[self.state[0][0]][it] for it in item_query]
        loss, pred = self.model.cal_loss([self.state[0][0]]*len(item_query), item_query, correct_query, self.know_map)

        pred_bin = np.where(pred > 0.5, 1, 0)
        ACC = np.sum(np.equal(pred_bin, correct_query)) / len(pred_bin) 
        try:
            AUC = roc_auc_score(correct_query, pred)
        except ValueError:
            AUC = -1

        self.model.init_stu_emb()
        return -loss, ACC, AUC, correct[-1]

    def precision(self, episode):
        return sum([i[1] for i in episode])

    def recall(self, episode, uid):
        return sum([i[1] for i in episode]) / len(self.rates[uid])

    def step_policy(self,policy):
        policy = policy[:self.args.T]
        rewards = []
        for action in policy:
            if action in self.rates[self.state[0][0]]:
                rewards.append(self.rates[self.state[0][0]][action])
            else:
                rewards.append(0)
        t = [[a,rewards[i],False] for i,a in enumerate(policy)]
        info = {"precision": self.precision(t),
                "recall": self.recall(t, self.state[0][0])}
        self.state[1].extend(t)
        return self.state,rewards,True,info



if __name__ == '__main__':
    args = {'T':10, 'data_path': './data/data/'}
    env(args)
