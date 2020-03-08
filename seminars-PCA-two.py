# YOU MAY NEED TO INSTALL THE FOLLOWING
#!pip install scikit-learn

#%matplotlib inline
#import numpy as np
#import matplotlib.pyplot as plt
#import seaborn as sns; sns.set()
#from sklearn.decomposition import PCA
#from sklearn.datasets import load_digits

#digits = load_digits()

pca = PCA().fit(digits.data)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');