# -*- coding: utf-8 -*-
"""cnn.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1o-EPqBTHVqVEMfvXDficqw8l_veQ7eWp
"""

import pandas as pd
import  numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

def load_data(fname):
    df = pd.read_csv(fname)
    return df

def load_preprocess_data(df, drop_threshold, n_knn):
  print('# of Patients')
  print(df['REF'].nunique())
  df['REF'] = df['REF'].apply(np.int64)

  #drop the columns that have more than X missing values
  columns_to_drop = ['DateOf1stSymptoms', 'firstDate',	'lastDate',	'medianDate']
  for col in df.columns:
    if col == 'Date_NIV' or col == 'Date_Critical':
      columns_to_drop.append(col)
    if df[col].isna().sum() > drop_threshold:
      columns_to_drop.append(col)

  df = df.drop(columns_to_drop, axis=1)

  #Divide the df in X and y
  X, y = df.loc[:, 'REF':'MITOS-stage'], df.loc[:, 'Evolution']

  #Encode Categorical Classes
  encoder_classes = [col for col in X.columns if X[col].dtypes == 'O']

  le = LabelEncoder()
  for class_enc in encoder_classes:
    label = le.fit_transform(df[class_enc])

    X = X.drop(class_enc, axis = 'columns')
    X[class_enc] = label

  # getting the cols that are not , 'firstDate', 'lastDate', 'medianDate'Object type
  num_cols = [col for col in X.columns if X[col].dtypes != 'O']

  # Changing the values of the target class (N,Y to 0,1)
  y = pd.get_dummies(df["Evolution"])
  y = y.drop(["N"], axis=1)
  
  # Rename the Column
  y = y.rename(columns={"Y": "Evolution"})

  #Missing Value Imputation
  imputer = KNNImputer(n_neighbors=n_knn, weights='uniform', metric='nan_euclidean')
  imputer.fit(X[num_cols])
  X[num_cols] = np.round(imputer.transform(X[num_cols]))
  if 'BMI' in num_cols:
    num_cols.remove('BMI')

  X[num_cols] =X[num_cols].round(decimals=1)
  X['BMI'] = X['BMI'].apply(np.float64)

  #plot target class distribution
  plot_dist(y)
    
  #ref dataframe
  ref_df = pd.DataFrame(X['REF'])
  X = X.drop('REF', axis=1)

  return X, y, ref_df

#Plot the target class distribution 
def plot_dist(y):
  evolution = y.loc[y['Evolution']==1, :].value_counts()
  no_evolution = y.loc[y['Evolution']==0, :].value_counts()
  df_plot = pd.DataFrame([evolution,no_evolution])
  df_plot.index = ['Evolution','No Evolution']

  # Bar plot
  df_plot.plot(kind='bar', title='90 Days Disease Progression');

def resample_data(X, y):
  # define oversampling strategy
  over = SMOTE(sampling_strategy = 0.8)
  #over = SMOTE()
  # fit and apply the transform
  X_sm, y_sm = over.fit_resample(X, y)
  #plot_dist(y_sm)
  # define undersampling strategy
  under = RandomUnderSampler()
  # fit and apply the transform
  X_sm, y_sm = under.fit_resample(X_sm, y_sm)
  #plot_dist(y_sm)

  return X_sm, y_sm
