from __future__ import division
#from loads import *
from timeit import default_timer as timer
from numba import cuda, float32
import numpy as np
import pickle
import math
#from progressbar import ProgressBar
import tqdm
import json

# Calulate Features
def calc_features(X):
    return X

# Predict
def forward(sample):
    return np.matmul(sample,W)

# Gradien Descent
def backward(y_actl, y_pred, sample):
    global W, LIST_W

    reduction = y_pred - y_actl
    gradient = reduction*np.transpose(sample)
    
    W = W*1.0 - 1.0*lr*gradient  
    LIST_W = np.append(LIST_W, W, 1)

# Calculate Loss
def loss(y_actl, y_pred):
    numfeatures = y_pred.shape[0]*1.0

    y_pred -= y_pred%10
    diff = abs(y_pred - y_actl)
    diff /= 10.0
    diff = diff*diff*1.0
    
    scalar_LOSS = np.sum(diff)
    scalar_LOSS /= (2.0*numfeatures)
    return scalar_LOSS



@cuda.jit
def stochastic(feat_train,y_train,weights):
    blockindex = cuda.grid(1)
    weight = weights[blockindex]
    maxsize = feat_train.shape[0]

    for i in range(SPB):
        index = i+blockindex*SPB
        if(index >= maxsize):
            return
        feature = feat_train[index]
        y_pred = 0.0
        for j in range(feature.shape[0]):
            y_pred += (feature[j]*weight[j])
        y_pred -= y_pred%10
        reduction = y_pred - y_train[index][0]
        for j in range(weight.shape[0]):
            weight[j] = weight[j]*1.0 - 1.0*lr*reduction*feature[j]



PARALLELIZE = True
TRAIN_LOSS = []
VAL_LOSS = []

# Threads per block
TPB = 1
# Samples per block
SPB = 1000

# Initialize
no_epochs = 100
lr = 0.001
W_size = 91

W = np.zeros((W_size,1))
LIST_W = np.zeros((W_size, 0))

# Import from loads.py
x_train = XTRAIN
y_train = YTRAIN
x_val = XVAL
y_val = YVAL
x_test = XTEST
y_test = YTEST

y_train = np.expand_dims(y_train, axis=1) 
y_val = np.expand_dims(y_val, axis=1) 
y_test = np.expand_dims(y_test, axis=1) 
print("===========================")
print("Features: ", W_size)
print("===========================")
print("Training Samples:   ", x_train.shape[0])
print("Validation Samples: ", x_val.shape[0])
print("===========================")
print("X_TRAIN: ", x_train.shape)
print("Y_TRAIN: ", y_train.shape)
print("X_VAL:   ", x_val.shape)
print("Y_VAL:   ", y_val.shape)
print("X_TEST:  ", x_test.shape)
print("Y_TEST:  ", y_test.shape)
print("===========================")

# Features
feat_train = calc_features(x_train)
feat_val = calc_features(x_val)
print('\n')

# NORMAL
def train():
    for epoch in range(no_epochs):
        print("EPOCH: ", epoch)
        for index, sample in enumerate(feat_train):
            if index%100==0:
                print("   SAMPLE: ", index)
            y_pred_train = forward(sample)[0]
            backward(y_train[index][0], y_pred_train, sample)
        # Calculate the model's prediction on training data
        y_pred_train = forward(feat_train)
        # Compute the loss on training data
        loss_train = loss(y_train, y_pred_train)
        TRAIN_LOSS.append(loss_train)
        
        # Compute loss on validation data
        y_pred_val = forward(feat_val)
        # Compute the loss on validation data
        loss_val = loss(y_val, y_pred_val)
        VAL_LOSS.append(loss_val)
        
# PARALLELIZED
def parallelTrain():
    global W,SPB
    threadsperblock = TPB
    blockspergrid = int(math.ceil(feat_train.shape[0] / SPB))
    feat_train_device = cuda.to_device(feat_train)
    y_train_device = cuda.to_device(y_train)
    weights = cuda.device_array((blockspergrid, feat_train.shape[1]))

    # Training
    for epoch in range(no_epochs):
        print("EPOCH: ", epoch)
        stochastic[threadsperblock, blockspergrid](feat_train_device,y_train_device,weights)
        
        # Take mean weight
        W = np.zeros((W_size, 1))
        weightsHost = weights.copy_to_host()
        for block in range(weightsHost.shape[0]):
            for index in range(weightsHost.shape[1]):
                W[index][0] += weightsHost[block][index]
        W /= (1.0*weightsHost.shape[0])
        
        # Calculate the model's prediction on training data
        y_pred_train = forward(feat_train)
        # Compute the loss on training data
        loss_train = loss(y_train, y_pred_train)
        TRAIN_LOSS.append(loss_train)
        
        # Compute loss on validation data
        y_pred_val = forward(feat_val)
        # Compute the loss on validation data
        loss_val = loss(y_val, y_pred_val)
        VAL_LOSS.append(loss_val)


start = timer()
if not PARALLELIZE:
    train()
else:
    parallelTrain()
time = timer() - start

print('\n\n')
print("===========================")
print('Final Validation Loss: {}'.format(VAL_LOSS[-1]))
print('Time (in seconds):    ', str(time))
print("===========================")

# Compute loss on test dataset
y_pred_test = np.matmul(x_test,W)
test_loss = loss(y_test, y_pred_test)
print('Test Loss: ', test_loss)
print("===========================")


# Dumping results
results = {}
results['parallelize'] = PARALLELIZE
results['tpb'] = TPB
results['spb'] = SPB
results['epochs'] = no_epochs
results['lr'] = lr

results['weights'] = np.transpose(W).tolist()
results['train_loss'] = TRAIN_LOSS
results['val_loss'] = VAL_LOSS

results['test_loss'] = test_loss

results['x_train'] = x_train.shape[0]
results['y_train'] = y_train.shape[0]
results['x_val'] = x_val.shape[0]
results['y_val'] = y_val.shape[0]
results['x_test'] = x_test.shape[0]
results['y_test'] = y_test.shape[0]

results['num_feat'] = x_train.shape[1]
results['time'] = time

#DATA_FILENAME='SGD_Results.json'
#with open(DATA_FILENAME, mode='r', encoding='utf-8') as feedsjson:
#    feeds = json.load(feedsjson)
#with open(DATA_FILENAME, mode='w', encoding='utf-8') as feedsjson:
#    feeds[str(len(feeds))] = results
#    json.dump(feeds, feedsjson)