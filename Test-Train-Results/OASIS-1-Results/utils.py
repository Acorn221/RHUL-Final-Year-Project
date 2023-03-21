import os
import math
import numpy as np

# The directory where the processed scans are stored
directory='C:/Active-Projects/RHUL-FYP/PROJECT/OASIS/1/processed_scans/'


def createFileName(id):
	return directory + id + ".png"

def convertToString(cdr):
	if str(cdr) == 'nan':
		return '0'
	else:
		return str(cdr)

def mmseConvert(x):
	if math.isnan(float(x)):
		return np.random.randint(25, 30)
	else:
		return float(x)

def genderToFloat(x):
	return 1 if x == 'M' else 0