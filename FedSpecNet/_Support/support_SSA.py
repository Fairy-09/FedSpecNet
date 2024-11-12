import numpy as np
# import matplotlib.pyplot as plt
import pandas as pd


def SSA(series, level):

	series = series - np.mean(series)  


	windowLen = level 
	seriesLen = len(series) 
	K = seriesLen - windowLen + 1  
	X = np.zeros((windowLen, K))    
	for i in range(K):
		X[:, i] = series[i:i + windowLen]   

	
	U, sigma, VT = np.linalg.svd(X, full_matrices=False)  

	for i in range(VT.shape[0]):
		VT[i, :] *= sigma[i]   
	A = VT   


	rec = np.zeros((windowLen, seriesLen))
	for i in range(windowLen):
		for j in range(windowLen - 1):
			for m in range(j + 1):
				rec[i, j] += A[i, j - m] * U[m, i]
			rec[i, j] /= (j + 1)
		for j in range(windowLen - 1, seriesLen - windowLen + 1):
			for m in range(windowLen):
				rec[i, j] += A[i, j - m] * U[m, i]
			rec[i, j] /= windowLen
		for j in range(seriesLen - windowLen + 1, seriesLen):
			for m in range(j - seriesLen + windowLen, windowLen):
				rec[i, j] += A[i, j - m] * U[m, i]
			rec[i, j] /= (seriesLen - j)

	
	res = pd.DataFrame(rec.T, columns=[f'rec_{i + 1}' for i in range(windowLen)])

	res.to_csv('output.csv', index=False)

	return rec

