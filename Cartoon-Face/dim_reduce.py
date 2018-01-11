from sklearn.decomposition import PCA 
from sklearn.decomposition import FastICA
from time import time
import pdb

def pca_decmp(X_train, X_test, n_comp):
	n_components = n_comp
	
	h,w = 48, 48
	pca = PCA(n_components=n_components, whiten=True).fit(X_train)

	
	eigenfaces = pca.components_.reshape((n_components, h, w))
	X_train_pca = pca.transform(X_train)
	X_test_pca = pca.transform(X_test)

	return(X_train_pca, X_test_pca, eigenfaces)

def ica_decmp(X_train, X_test, n_comp):
	n_components = n_comp
	
	h,w = 48, 48
	ica = FastICA(n_components=n_components, max_iter=1000, whiten=True).fit(X_train)
	
	icafaces = ica.components_.reshape((n_components, h, w))
	X_train_ica = ica.transform(X_train)
	X_test_ica = ica.transform(X_test)

	return(X_train_ica, X_test_ica, icafaces)

    
