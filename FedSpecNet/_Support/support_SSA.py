import numpy as np
# import matplotlib.pyplot as plt
import pandas as pd


def SSA(series, level):

	windowLen = level
	seriesLen = len(series)
	K = seriesLen - windowLen + 1
	X = np.zeros((windowLen, K))
	for i in range(K):
	    X[:, i] = series[i:i + windowLen]
	
	U, sigma, VT = np.linalg.svd(X)
	
	rec = np.zeros((level, seriesLen))
	mask = np.full((seriesLen,), 1 / level)
	for j in range(level-1):
	    mask[j] *= level / (j+1)
	    mask[-j - 1] *= level / (j+1)
	
	for i in range(level):
	    A = np.matmul(U[:, i:i+1], VT[i:i+1, :])
	    A *= sigma[i]
	    x = np.zeros((seriesLen,))
	    for j in range(level):
	       temp_x = np.pad(A[j, :], (j, level-1-j), 'constant', constant_values=0.)
	       x += temp_x
	    x *= mask
	    rec[i, :] = x
	return rec



