import numpy as np
import math
from numba import cuda, float32

def distance(a, b):
    return np.linalg.norm(a-b)


def kmean(X, k = 1, n_iter = 25):
    n_clusters = k
    n_iterations = n_iter
    n_data_points = X.shape[0]
    n_dimensions = X.shape[1]

    labels = np.zeros(n_data_points)
    random_centers = np.random.choice(n_data_points, size=n_clusters, replace=False)
    random_centers = [720,2001]
    centers = X[random_centers]
    
    print(centers.shape)
    print(labels.shape)
    
    for count in range(n_iterations):
        newCenters = np.zeros(centers.shape)
        clusterFreq = np.zeros(n_clusters)

        for index in range(n_data_points):
            distances = np.zeros(n_clusters)
            for label in range(n_clusters):
                distances[label] = distance(X[index],centers[label])

            cluster = 0
            for label in range(n_clusters):
                if distances[label] < distances[cluster]:
                    cluster = label

            labels[index] = cluster
            newCenters[cluster] += X[index]
            clusterFreq[cluster] += 1.0
        
        for index in range(n_clusters):
            centers[index] = newCenters[index]/clusterFreq[index]

        print(count)
    print("DONE")

    return labels

SPB = 100

@cuda.jit
def kmeans_kernel():
    blockindex = cuda.grid(1)
    print(blockindex)
#    maxsize = data_points.shape[0]
#    n_clusters = centers.shape[0]
#    
#    for i in range(SPB):
#        index = i+blockindex*SPB
#        if(index >= maxsize):
#            return
#        sample = data_points[index]
#            
#        print("here")
#        print(sample.shape)
#        for label in range(n_clusters):
#            distances[label] = distance(X[index],centers[label])
#
#        cluster = 0
#        for label in range(n_clusters):
#            if distances[label] < distances[cluster]:
#                cluster = label

import numpy as np
from feature import *
#from utils import *

data = readfiles('dataset')
bow = BagOfWordsFeatureExtractor()
bow.preprocess(data)

X_data_bow = bow.extract(data)
#labels = kmeanparallel(X_data_bow, k=2, n_iter=2)
k=2
n_iter=2
X = X_data_bow
#def kmeanparallel(X, k = 1, n_iter = 25):

n_clusters = k
n_iterations = n_iter
n_data_points = X.shape[0]
n_dimensions = X.shape[1]

labels = np.zeros(n_data_points)
random_centers = np.random.choice(n_data_points, size=n_clusters, replace=False)
random_centers = [720,2001]
centers = X[random_centers]

threadsperblock = 1
blockspergrid = int(math.ceil(n_data_points/SPB))
#centers_device = cuda.to_device(centers)
#data_points = cuda.to_device(X)
#means = cuda.device_array((blockspergrid*n_clusters, centers.shape[1]))
#freq = cuda.device_array((blockspergrid*n_clusters, centers.shape[1]))

print(threadsperblock,blockspergrid)
print("codebase1")
#    kmeans_kernel[threadsperblock, blockspergrid](data_points,centers_device,means,freq)
kmeans_kernel[threadsperblock, blockspergrid]()
print("codebase2")

for count in range(n_iterations):
    newCenters = np.zeros(centers.shape)
    clusterFreq = np.zeros(n_clusters)
    for index in range(n_data_points):
        distances = np.zeros(n_clusters)
        for label in range(n_clusters):
            distances[label] = distance(X[index],centers[label])

        cluster = 0
        for label in range(n_clusters):
            if distances[label] < distances[cluster]:
                cluster = label

        labels[index] = cluster
        newCenters[cluster] += X[index]
        clusterFreq[cluster] += 1.0
    
    for index in range(n_clusters):
        centers[index] = newCenters[index]/clusterFreq[index]

    print(count)
print("DONE")

#    return labels

