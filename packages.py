#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 12:38:34 2018

@author: lunar
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import seaborn as sns
import re
import matplotlib
from collections import Counter
import time
import json
import pickle
from scipy.stats import pearsonr

#preprocessing
from sklearn.preprocessing import MultiLabelBinarizer, label_binarize, LabelEncoder, StandardScaler
from sklearn import decomposition

# utils for validation of a model and hyberparameters search
from sklearn.model_selection import (KFold, train_test_split,
cross_validate, cross_val_score, GridSearchCV, LeaveOneOut) 

from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn import pipeline

#For classification task we will first try using simple tree algorithms
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn import svm
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, ExtraTreesRegressor

from xgboost import XGBClassifier, XGBRegressor