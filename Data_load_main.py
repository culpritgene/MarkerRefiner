#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 12:29:43 2018

@author: lunar
"""
import os
os.chdir('/home/lunar/Desktop/research_project/results/Pipeline')

from packages import *
from ml_utils import *
from pipeline import *
from data_preparation import *

warnings.simplefilter('ignore')

path_to_data = '/home/lunar/Desktop/research_project/Data/Liver/'

tumors = pd.read_csv(path_to_data+'tumors_liver_graded_etc_non_dupl_2.csv', index_col=0)
tumors.index = tumors.donor
tumors = tumors[tumors['case']==1]
X, y = tumors.ix[:,:-9], tumors.tumour_grade

## drop outliers
## Already selected for Liver dataset 
outs_1 = "DO45139 DO45173 DO50859 DO45096 DO50793 DO45281 DO45141".split()
outs_II = 'DO45237 DO23523 DO45189 DO23530 DO45277 DO45209 DO45211'.split()
outliers = outs_1 + outs_II

PcX = pca_hueing(X,y, name='Total)
outliers = outliers_pca(PcX)

tumors = tumors.T.drop(outliers, axis=1).T
## encode data, select classes
select_data(tumors, by='tumour_grade', n_classes=None, list_of_classes=None):

##










