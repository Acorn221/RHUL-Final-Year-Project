import os
import math
import numpy as np

# The directory where the processed scans are stored
directory='./processed_scans/'


def createFileName(id):
	return directory + id + ".png"

def convertToString(cdr):
	if str(cdr) == 'nan':
		return '0'
	else:
		return str(cdr)

# Not all MMSE scores are available, so we need to fill in the missing values, and assume that the missing values are in the range of 25-30
def mmseConvert(x):
	if math.isnan(float(x)):
		return np.random.randint(25, 30)
	else:
		return float(x)

def genderToFloat(x):
	return 1 if x == 'M' else 0

# The CDR scores are in the range of 0-2, with 0.5 increments. We need to convert them to one-hot arrays
def cdr_formatting(s):

	values = ['0.0', '0.5', '1.0', '2.0']

	one_hot_array = np.zeros(len(values), dtype=int)

	index = values.index(str(float(s)))

	one_hot_array[index] = 1

	return one_hot_array