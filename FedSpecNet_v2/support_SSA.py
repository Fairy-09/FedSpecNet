import numpy as np
# import matplotlib.pyplot as plt

#data:(9608,1)
def SSA(series, level):  	## 输入时间序列数据，嵌入窗口长度
	# series = 0
	# series = series - np.mean(series)  # 中心化(非必须)
	original_mean = np.mean(series)  # 保存原始数据的平均值###
	series = series - original_mean  # 中心化

	# step1 嵌入
	windowLen = level  # 嵌入窗口长度4
	seriesLen = len(series)  # 序列长度9608
	K = seriesLen - windowLen + 1   #K:(9605,)  嵌入后矩阵列数
	series = series.flatten()
	X = np.zeros((windowLen, K))    #X:(4,9605)	初始化零矩阵
	for i in range(K):
		X[:, i] = series[i:i + windowLen]  
		##将长度为 `windowLen` 的滑动窗口在原始时间序列`series` 上进行滑动，提取子序列并将其作为矩阵X的列

	# step2: svd奇异值分解， U和sigma已经按升序排序
	U, sigma, VT = np.linalg.svd(X, full_matrices=False)   #U:(4,4)

	for i in range(VT.shape[0]):
		VT[i, :] *= sigma[i]
	A = VT    #A:(4,9605),VT:(4,9605)   ## A=VT*Σ

	# 重构时间序列  ## 这部分看论文
	rec = np.zeros((windowLen, seriesLen))    ## 初始化零矩阵，用于存储重构的时间序列
	for i in range(windowLen):
		for j in range(windowLen - 1):
			for m in range(j + 1):
				rec[i, j] += A[i, j - m] * U[m, i]
			rec[i, j] /= (j + 1)						## 过去值的加权平均
		for j in range(windowLen - 1, seriesLen - windowLen + 1):
			for m in range(windowLen):
				rec[i, j] += A[i, j - m] * U[m, i]
			rec[i, j] /= windowLen						## 固定窗口长度内值的加权平均
		for j in range(seriesLen - windowLen + 1, seriesLen):
			for m in range(j - seriesLen + windowLen, windowLen):
				rec[i, j] += A[i, j - m] * U[m, i]
			rec[i, j] /= (seriesLen - j)                ## 过去值的加权平均
	# for i in range(windowLen):
	# 	rec[i, :] += original_mean
	return rec

'''
原始时间序列 -> 中心化 -> 轨迹矩阵（嵌入）-> SVD -> 重构 -> 重构时间序列
'''
# rrr = np.sum(rec, axis=0)  # 选择重构的部分，这里选了全部
#
# plt.figure()
# for i in range(10):
# 	ax = plt.subplot(5, 2, i + 1)
# 	ax.plot(rec[i, :])
#
# plt.figure(2)
# plt.plot(series)
# plt.show()