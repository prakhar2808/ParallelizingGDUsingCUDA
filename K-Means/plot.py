import matplotlib.pyplot as plt
import pandas as pd

from sklearn.decomposition import PCA as sklearnPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.datasets.samples_generator import make_blobs

from pandas.tools.plotting import parallel_coordinates

pca = sklearnPCA(n_components=2) #2-dimensional PCA
transformed = pd.DataFrame(pca.fit_transform(X_data_bow))

plt.scatter(transformed[labels==0][0], transformed[labels==0][1], label='Class 1', c='red')
plt.scatter(transformed[labels==1][0], transformed[labels==1][1], label='Class 2', c='blue')

plt.legend()
plt.show()