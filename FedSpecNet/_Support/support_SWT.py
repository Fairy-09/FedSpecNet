import pywt
from matplotlib import pyplot as plt

def swt_decom(data, wavefunc, lv):
	
	coeffs_list = []  	
	data_to_decom = data
	A = None
	
	for i in range(lv): 	
		[(A, D)] = pywt.swt(data_to_decom, wavefunc, level=1, axis=0)  	
		data_to_decom = A 	
		coeffs_list.insert(0, D) 	
	coeffs_list.insert(0, A) 	
	
	return coeffs_list



def swt_decom_high(data, wavefunc, lv):
	coeffs_list = []
	data_to_decom = data
	D = None
	
	for i in range(lv):
		[(A, D)] = pywt.swt(data_to_decom, wavefunc, level=1, axis=0)
		data_to_decom = D
		coeffs_list.insert(0, A)
	coeffs_list.insert(0, D)
	
	return coeffs_list



def iswt_decom(data, wavefunc):
	
	y = data[0]
	for i in range(len(data) - 1):
		y = pywt.iswt([(y, data[i+1])], wavefunc)
	return y



def iswt_decom_high(data, wavefunc):
	y = data[0]
	for i in range(len(data) - 1):
		y = pywt.iswt([(data[i + 1], y)], wavefunc)
	return y
	


def swpt_decom_V1(data, wavefunc, lv):
	data_to_decom = data
	[(A, D)] = pywt.swt(data_to_decom, wavefunc, level=1, axis=0)
	coeffs_list_low = swt_decom(A, wavefunc, lv - 1)
	coeffs_list_high = swt_decom_high(D, wavefunc, lv - 1)
	coeffs_list = coeffs_list_low + coeffs_list_high
	return coeffs_list


def iswpt_decom_V1(data, wavefunc):
	coeffs_list_low = data[:(len(data))//2]
	coeffs_list_high = data[:(len(data))//2]
	coeffs_low = iswt_decom(coeffs_list_low, wavefunc)
	coeffs_high = iswt_decom_high(coeffs_list_high, wavefunc)
	original = iswt_decom([coeffs_low, coeffs_high], wavefunc)
	return original


def swpt_decom_V2(data, wavefunc, lv):
	data_to_decom = data
	[(A, D)] = pywt.swt(data_to_decom, wavefunc, level=1, axis=0)
	coeffs_list_low = swt_decom(A, wavefunc, lv - 1)
	coeffs_list_high = swt_decom(D, wavefunc, lv - 1)
	coeffs_list = coeffs_list_low + coeffs_list_high
	return coeffs_list


def iswpt_decom_V2(data, wavefunc):
	coeffs_list_low = data[:(len(data))//2]
	coeffs_list_high = data[:(len(data))//2]
	coeffs_low = iswt_decom(coeffs_list_low, wavefunc)
	coeffs_high = iswt_decom(coeffs_list_high, wavefunc)
	original = iswt_decom([coeffs_low, coeffs_high], wavefunc)
	return original





