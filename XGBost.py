#%% Importing necessary libraries
import numpy as np
import pandas as pd
import xgboost

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score

import matplotlib.pyplot as plt

#%% Read in csvs
df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

df_train.info()
df_train.describe()
df_train.isnull().sum()

# %% Converting to appropriate data type and dropping user_id column
df_train['label'] = df_train['label'].astype(object)
df_train = df_train.drop(columns = ['user_id'])

# %% Checking label distribution/ class imbalance
group = df_train.groupby('label')
group.size()
#df_train.size()

#%%
prop = [group.size()[0]/group.size().sum(),
group.size()[1]/group.size().sum()]
prop

#%% Data Pre-processing
df_train = df_train.fillna(df_train.median()) #Median imputation
df_train = df_train.loc[:,(df_train.min() != df_train.max())] # Removing single-valued variables

#%%
from sklearn.preprocessing import normalize
df_train = normalize(df_train) # Normalizing predictors

# %% Separating response from predictors
X  = df_train.drop(columns = ['label'])
y = df_train['label']
#%% Tackling class imbalance
from imblearn.over_sampling import SMOTE
oversample = SMOTE()
X,y = oversample.fit_resample(X, y)


#%% Partitioning df_train into test and train for local training and testing
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=.2)
X_train.shape, X_test.shape




#%% Creating the XGB model

