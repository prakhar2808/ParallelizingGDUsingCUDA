from __future__ import division
#from loads import *
from timeit import default_timer as timer
from numba import cuda, float32
import numpy as np
import pickle
import math
from progressbar import ProgressBar
import tqdm

# Calulate Features
def calc_features(X):
    return X


@cuda.jit
def matmul_kernel(A, B, C):
    """
    Perform matrix multiplication of C = A * B
    Each thread computes one element of the result matrix C
    """
    # Define an array in the shared memory
    # The size and type of the arrays must be known at compile time
    sA = cuda.shared.array(shape=(TPB, TPB), dtype=float32)
    sB = cuda.shared.array(shape=(TPB, TPB), dtype=float32)
    
    x, y = cuda.grid(2)
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    
    if x >= C.shape[0] and y >= C.shape[1]:
        # Quit if (x, y) is outside of valid C boundary
        return

    # Each thread computes one element in the result matrix.
    # The dot product is chunked into dot products of TPB-long vectors.
    tmp = 0.
    for i in range(int(A.shape[1] / TPB)):
        # Preload data into shared memory
        sA[tx, ty] = A[x, ty + i * TPB]
        sB[tx, ty] = B[tx + i * TPB, y]

        # Wait until all threads finish preloading
        cuda.syncthreads()

        # Computes partial product on the shared memory
        for j in range(TPB):
            tmp += sA[tx, j] * sB[j, ty]

        # Wait until all threads finish computing
        cuda.syncthreads()

    C[x, y] = tmp


def matmul(A, B, matmultype='forward'):
    global PARALLELIZE
    if not PARALLELIZE:
        # NORMAL
        gradient = np.zeros((A.shape[0], B.shape[1]))
        for i in range(A.shape[0]):
           for j in range(B.shape[1]):
               for k in range(A.shape[1]):
                   gradient[i][j] += A[i][k]*B[k][j]
        return gradient
    
    # PARALLELIZED
    global global_feat
    global global_feat_val
    global global_feat_transpose
    B_global_mem = cuda.to_device(B)
    C_global_mem = cuda.device_array((A.shape[0], B.shape[1]))

    # Configure the blocks
    threadsperblock = (TPB, TPB)
    blockspergrid_x = int(math.ceil(A.shape[0] / threadsperblock[1]))
    blockspergrid_y = int(math.ceil(B.shape[1] / threadsperblock[0]))
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    # Start the kernel
    if matmultype=='forward':
        matmul_kernel[blockspergrid, threadsperblock](global_feat, B_global_mem, C_global_mem)
    elif matmultype=='validation':
        matmul_kernel[blockspergrid, threadsperblock](global_feat_val, B_global_mem, C_global_mem)
    elif matmultype=='backward':
        matmul_kernel[blockspergrid, threadsperblock](global_feat_transpose, B_global_mem, C_global_mem)

    res = C_global_mem.copy_to_host()
    return res


# Predict
def forward(sample, matmultype='forward'):
    global W
    return matmul(sample,W,matmultype=matmultype)

# Gradien Descent
def backward(y_actl, y_pred, sample):
    global W, LIST_W
    reduction = y_pred - y_actl
    
    samples = sample.shape[0]*1.0
    gradient = matmul(np.transpose(sample),reduction,matmultype='backward')
    gradient /= samples
    
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




PARALLELIZE = True
TRAIN_LOSS = []
VAL_LOSS = []

# Threads per block
TPB = 1

# Initialize
no_epochs = 10000 
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
global_feat = cuda.to_device(feat_train)
global_feat_val = cuda.to_device(feat_val)
global_feat_transpose = cuda.to_device(np.transpose(feat_train))
print('\n')

#kernprof -l <file>.py
#python -m line_profiler <file>.py.lprof
#@profile
def train():
    # Training
    for epoch in range(no_epochs):
        if epoch%10 == 0:
            print("EPOCH: ", epoch)
        
        # Calculate the model's prediction on training data
        y_pred_train = forward(feat_train)
        # Compute the loss on training data
        loss_train = loss(y_train, y_pred_train)
        TRAIN_LOSS.append(loss_train)
        
        # Compute loss on validation data
        y_pred_val = forward(feat_val, matmultype='validation')
        # Compute the loss on validation data
        loss_val = loss(y_val, y_pred_val)
        VAL_LOSS.append(loss_val)
        
        # Train the linear regression model, using gradient descent
        backward(y_train, y_pred_train, feat_train)
    

start = timer()
train()
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

# picklename = 'testloss.pkl'
# val_loss = open(picklename, 'wb')
# pickle.dump(VAL_LOSS, val_loss)

# picklename = 'trainloss.pkl'
# train_loss = open(picklename, 'wb')
# pickle.dump(TRAIN_LOSS, train_loss)

# Dumping results
results = {}
results['parallelize'] = PARALLELIZE
results['tpb'] = TPB
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
#
#DATA_FILENAME='BGD_Results.json'
#with open(DATA_FILENAME, mode='r', encoding='utf-8') as feedsjson:
#    feeds = json.load(feedsjson)
#with open(DATA_FILENAME, mode='w', encoding='utf-8') as feedsjson:
#    feeds[str(len(feeds))] = results
#    json.dump(feeds, feedsjson)