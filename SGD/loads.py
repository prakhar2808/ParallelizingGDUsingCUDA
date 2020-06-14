import numpy as np
import pandas as pd
from subprocess import check_output
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn import datasets, svm, metrics
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.utils import resample
import itertools
from sklearn.model_selection import GridSearchCV
from sklearn.manifold import TSNE

print("Loading Data..")

df = pd.read_csv('/home/navpun31/Documents/CUDA/year_prediction.csv')
df_test = pd.read_csv('/home/navpun31/Documents/CUDA/year_prediction_test.csv')

# Group release years into decades
df['label'] = df.label.apply(lambda year : year-(year%10))
df_test['label'] = df_test.label.apply(lambda year : year-(year%10))

#Plot Training Data
#sns.countplot(y="label", data=df)
#plt.xlabel("Audio samples")
#plt.ylabel("Release Decade")
#plt.title("Samples in the dataset/release decade")

print("Training : (Samples, Features) {}".format(df.iloc[:,1:].shape))
print("Test     : (Samples, Features) {}".format(df_test.iloc[:,1:].shape))

'''
SCALING FEATURES
After scaling these features using min-max scaling, each feature is reduced to a range of 0 to 1 
'''
df.iloc[:,1:91] = (df.iloc[:,1:91]-df.iloc[:,1:91].min())/(df.iloc[:,1:91].max() - df.iloc[:,1:91].min())
df_test.iloc[:,1:91] = (df_test.iloc[:,1:91]-df_test.iloc[:,1:91].min())/(df_test.iloc[:,1:91].max() - df_test.iloc[:,1:91].min())


samplesize = 50000
#Training and Validation Data
df_sampled = df
df_sampled = df_sampled.sample(samplesize)

df_sampled = shuffle(df_sampled)
df_train, df_val = train_test_split(df_sampled, test_size=0.2)

XTRAIN = df_train.iloc[:,1:].values 
YTRAIN = df_train.iloc[:,0].values
XVAL = df_val.iloc[:,1:].values 
YVAL = df_val.iloc[:,0].values
print("XTRAIN ", XTRAIN.shape, ", YTRAIN ", YTRAIN.shape)
print("XVAL   ", XVAL.shape,   ", YVAL   ", YVAL.shape)

#Test Data
df_sampled_test = df_test
df_sampled_test = df_sampled_test.sample(int(samplesize/5))
XTEST = df_sampled_test.iloc[:,1:].values
YTEST = df_sampled_test.iloc[:,0].values
print("XTEST  ", XTEST.shape, ", YTEST ", YTEST.shape)
