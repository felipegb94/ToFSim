#### Python imports
import math
#### Library imports
import numpy as np
import scipy as sp
from scipy import fftpack 
from scipy import signal
from scipy import linalg
from scipy import interpolate



def ScaleAreaUnderCurve(x, dx=0., desiredArea=1.):
	"""ScaleAreaUnderCurve: Scale the area under the curve x to some desired area.
	
	Args:
	    x (TYPE): Discrete set of points that lie on the curve. Numpy vector
	    dx (float): delta x. Set to 1/length of x by default.
	    desiredArea (float): Desired area under the curve.
	
	Returns:
	    numpy.ndarray: Scaled vector x with new area.
	"""
	#### Validate Input
	# assert(UtilsTesting.IsVector(x)),'Input Error - ScaleAreaUnderCurve: x should be a vector.'
	#### Calculate some parameters
	N = x.size
	#### Set default value for dc
	if(dx == 0): dx = 1./float(N)
	#### Calculate new area
	oldArea = np.sum(x)*dx
	y = x*desiredArea/oldArea
	#### Return scaled vector
	return y 


def ApplyKPhaseShifts(x, shifts):
	"""ApplyPhaseShifts: Apply phase shift to each vector in x. 
	
	Args:
	    x (np.array): NxK matrix
	    shifts (np.array): Array of dimension K.
	
	Returns:
	    np.array: Return matrix x where each column has been phase shifted according to shifts. 
	"""
	K = 0
	if(type(shifts) == np.ndarray): K = shifts.size
	elif(type(shifts) == list): K = len(shifts) 
	else: K = 1
	for i in range(0,K):
		x[:,i] = np.roll(x[:,i], int(round(shifts[i])))

	return x

def GetCorrelationFunctions(ModFs, DemodFs, dt=None):
	"""GetCorrelationFunctions: Calculate the circular correlation of all modF and demodF.
	
	Args:
	    corrFAll (numpy.ndarray): Correlation functions. N x K matrix.
	
	Returns:
	    np.array: N x K matrix. Each column is the correlation function for the respective pair.
	"""
	#### Reshape to ensure needed dimensions
	if(ModFs.ndim == 1): ModFs = ModFs.reshape((ModFs.shape[0], 1))
	if(DemodFs.ndim == 1): DemodFs = DemodFs.reshape((DemodFs.shape[0], 1))
	## Assume that the number of elements is larger than the number of coding pairs, i.e. rows>cols
	if(ModFs.shape[0] < ModFs.shape[1]): ModFs = ModFs.transpose()
	if(DemodFs.shape[0] < DemodFs.shape[1]): DemodFs = DemodFs.transpose()
	#### Verify Inputs
	assert(ModFs.shape == DemodFs.shape), "Input Error - PlotCodingScheme: ModFs and \
	DemodFs should be the same dimensions."
	#### Declare some parameters
	(N,K) = ModFs.shape
	#### Get dt
	if(dt == None): dt = 1./N
	#### Allocate the correlation function matrix
	CorrFs = np.zeros(ModFs.shape)
	#### Get correlation functions
	for i in range(0,K):
		CorrFs[:,i] = np.fft.ifft(np.fft.fft(ModFs[:,i]).conj() * np.fft.fft(DemodFs[:,i])).real
	#### Scale by dt
	CorrFs = CorrFs*dt	
	return CorrFs


def NormalizeBrightnessVals(BVals):
	## Normalized correlation functions, zero mean, unit variance. We have to transpose so that broadcasting works.
	NormBVals = (BVals.transpose() - np.mean(BVals, axis=1)) / np.std(BVals, axis=1) 
	# Transpose it again so that it has dims NxK
	NormBVals = NormBVals.transpose()
	return NormBVals