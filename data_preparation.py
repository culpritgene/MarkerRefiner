#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 12:40:52 2018

@author: lunar
"""
import os
os.chdir('/home/lunar/Desktop/research_project/results/Pipeline')
from packages import *

def select_data(tumors, by='tumour_grade', n_classes=None, list_of_classes=None):
    ## append classes names to sample names
    tumors.index = ['{}{}'.format(a,b) for a,b in zip(tumors.index, tumors[by].values)]

    if list_of_classes:
        tumors = tumors.loc[tumors['column_name'].isin(list_of_classes)]
    ### select n classes
    if n_classes:
        selected_classes = tumors[by].value_counts().index[:n_classes]
        tumors = tumors.loc[tumors['column_name'].isin(selected_classes)]
   
    X_f, y_f = tumors.ix[:,:-9], tumors[by]
    #### merge some classes with each other
    y_f = y_f.replace({'I':'I', 'I-II':'I', 'II-I':'II', 'II':'II', 'II-III':'III', 'III':'III', 'IV':'III'}, inplace=True)
     
    print('Set of {} clsasses: ;'.format(len(y_f.unique()), y_f.unique()))

    ynf_enc = LabelEncoder()
    ynf_enc.fit(y_f)
    return X_f, y_f



