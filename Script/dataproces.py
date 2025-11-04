import time
import pickle
import pandas as pd
import numpy as np
import os
import csv
import sys
import matplotlib.pyplot as plt
from matrix_cov import *
import threading
from concurrent.futures import ThreadPoolExecutor
import math
import torch
import random
class data_process():

    def __init__(self,genenotype_file,trait_file,save_dir):
        pos_list = pd.read_csv(r"./snp.txt",header = None)
        self.pos_list = pos_list.iloc[:,0].to_list()
        self.geneotype_path = genenotype_file
        self.trait_path = trait_file
        self.save_path = save_dir
        self.get_row()
        self.convert_trait()
    def get_row(self):
        skipped = []
        csv.field_size_limit(500 * 1024 * 1024)
        with open(self.geneotype_path, 'r') as csvfile:
            reader = csv.reader(csvfile)
            for i, row in enumerate(reader):
                if row[0].strip()[:2] == '##':
                    skipped.append(i)
            self.skipped = skipped
    def convert_trait(self):
        print("*---------------convert_trait---------------*")
        csv = pd.read_csv(self.trait_path)
        sample_list = csv.iloc[:,0]
        df = np.array(csv['type']).tolist()
        flag_p = 1
        flag_n = 1
        p_trait = os.path.join(self.save_path,'p_trait.txt')
        n_trait = os.path.join(self.save_path,'n_trait.txt')
        p_trait_dict = open(p_trait,'w')
        n_trait_value = open(n_trait,'w')
        for column in csv.columns[1:]:
            try:
                csv[column].astype('float')
                print(column)
                if flag_n:
                    flag_n = 0
                    end = column
                max,min = csv[column].max(),csv[column].min()
                csv[column] = (csv[column] - min) / (max - min)
                n_trait_value.write(f'{column};')
                n_trait_value.write(f'max;{max};min;{min}')
                n_trait_value.write('\n')

            except ValueError:
                
                    if flag_p:
                        start = column
                        flag_p = 0
                    map_dict = {}
                    array = np.array(csv[column])
                    target_set = list(set(array.tolist()))
                    value = [ i for i in range(0,len(target_set))]
                    for key,value in zip(target_set,value):
                        map_dict[key] = value
                    csv[column] = csv[column].map(map_dict)

                    p_trait_dict.write(f'{column}:')
                    p_trait_dict.write(f'{map_dict}')
                    p_trait_dict.write('\n')
        start = list(csv.columns).index(start)
        end = list(csv.columns).index(end)
        p_trait = csv.iloc[:,start:end]
        n_trait = csv.iloc[:,end:csv.shape[1]]
        n_trait['type'] = df
        p_trait.insert(0,'acid',sample_list)
        n_trait.insert(0,'acid',sample_list)
        self.p_trait_path = os.path.join(self.save_path,'p_value.csv')
        self.n_trait_path = os.path.join(self.save_path,'n_trait.csv')
        p_trait.to_csv(os.path.join(self.save_path,'p_value.csv'))
        n_trait.to_csv(os.path.join(self.save_path,'n_trait.csv'))
        p_trait_dict.close()
        n_trait_value.close()
        print("convert trait has finished!")
        print("*---------------convert_trait---------------*")
    def get_data(self,dataframe,trait,data_type):
        data_marix = np.array(dataframe)
        self.data_marix = data_marix
        self.sample_list = list(dataframe.index)
        data =[]
        label = []
        for sample in range(data_marix.shape[0]):
            if np.isnan(np.array(trait[f'{self.sample_list[sample]}'])):
                continue
            else:
                one_hot = np.zeros((1,data_marix[sample].shape[0],3))
                for snp in range(len(data_marix[sample])):
                    if data_marix[sample][snp] == '1|1':
                        one_hot[0,snp,0] = 1
                        one_hot[0,snp,1] = 1
                        one_hot[0,snp,2] = 0
                    elif data_marix[sample][snp] == '0|1':
                        one_hot[0,snp,0] = 1
                        one_hot[0,snp,1] = 0
                        one_hot[0,snp,2] = 1
                    else:
                        one_hot[0,snp,0] = 0
                        one_hot[0,snp,1] = 1
                        one_hot[0,snp,2] = 1
                one_hot = np.resize(one_hot,(206,206,3))
                target = np.array(trait[f'{self.sample_list[sample]}'].astype('float64'))
                data.append(torch.from_numpy(one_hot))
                label.append(torch.from_numpy(target))
        print(f'{data_type} dataset already completed!')
        print(len(data))
        return data,label
    
    def to_dataset(self,trait_for_epoch,if_train = True,if_n_trait =  True):
        trait_path = self.n_trait_path if if_n_trait  else self.p_trait_path
        data_type = 'train' if if_train  else 'test'
        skip = self.skipped
        trait = pd.read_csv(trait_path)
        trait = trait.set_index('acid')
        trait = trait[pd.notna(trait[trait_for_epoch[0]])]
        print(f"trait shape {trait.shape}")
        train_samples = trait[trait['type'] == 'train'].index.to_list()
        test_samples = trait[trait['type'] == 'test'].index.to_list()
        trait = trait[trait_for_epoch].transpose()
        print(f"trait shape {trait.shape}")

        df = pd.read_csv(self.geneotype_path, sep=r"\s+", skiprows=skip)
        df['ID'] = df['#CHROM'].map(str) + '_' + df['POS'].map(
            int).map(str)
        df = df.drop(columns=[
            'QUAL', 'FILTER', 'INFO', 'FORMAT', '#CHROM', 'POS', 'REF', 'ALT'
        ])

        df = df.set_index('ID')
        for i in df.columns:
            df[i] = df[i].str[:3]
            
        df = df.transpose()
        print(f"df shape {df.shape}")
        train_df = df[self.pos_list].loc[train_samples]
        test_df = df[self.pos_list].loc[test_samples]
        print(train_df.shape,test_df.shape)
        train_data,train_label = self.get_data(train_df,trait,data_type='train')
        test_data,test_label = self.get_data(test_df,trait,data_type='test')

        return train_data,train_label,test_data,test_label
