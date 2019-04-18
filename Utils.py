#### Python imports
import math
#### Library imports
import numpy as np
import scipy as sp
from scipy import stats
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


def ScaleMod(ModFs, tau=1., pAveSource=1.):
	"""ScaleMod: Scale modulation appropriately given the beta of the scene point, the average
	source power and the repetition frequency.
	
	Args:
	    ModFs (np.ndarray): N x K matrix. N samples, K modulation functions
	    tau (float): Repetition frequency of ModFs 
	    pAveSource (float): Average power emitted by the source 
	    beta (float): Average reflectivity of scene point

	Returns:
	    np.array: ModFs 
	"""
	(N,K) = ModFs.shape
	dt = tau / float(N)
	eTotal = tau*pAveSource # Total Energy
	for i in range(0,K): 
		ModFs[:,i] = ScaleAreaUnderCurve(x=ModFs[:,i], dx=dt, desiredArea=eTotal)

	return ModFs


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


def ComputeBrightnessVals(ModFs, DemodFs, depths=None, pAmbient=0, beta=1, T=1, tau=1, dt=1, gamma=1):
	"""ComputeBrightnessVals: Computes the brightness values for each possible depth.
	
	Args:
	    ModFs (np.ndarray): N x K matrix. N samples, K modulation functions
	    DemodFs (np.ndarray): N x K matrix. N samples, K demodulation functions
	    tau (float): Repetitiion period of ModFs and DemodFs
	    pAmbient (float): Average power of the ambient illumination component
	    beta (float): Reflectivity to be used
	    T (float): 
	Returns:
	    np.array: ModFs 
	"""
	(N,K) = ModFs.shape
	if(depths is None): depths = np.arange(0, N, 1)
	depths = np.round(depths)
	## Calculate correlation functions (integral over 1 period of m(t-phi)*d(t)) for all phi
	CorrFs = GetCorrelationFunctions(ModFs,DemodFs,dt=dt)
	## Calculate the integral of the demodulation function over 1 period
	kappas = np.sum(DemodFs,0)*dt
	## Calculate brightness values
	BVals = (gamma*beta)*(T/tau)*(CorrFs + pAmbient*kappas)
	## Return only the brightness vals for the specified depths
	BVals = BVals[depths,:]

	return (BVals)

def GetClippedBSamples(nSamples, BMean, BVar):
	"""GetClippedBSamples: Draw N brightness samples from the truncated multivariate gaussian dist 
	with mean BVal and Covariance Sigma=diag(NoiseVar)
	Args:
	    nSamples (int): Number of samples to draw.
	    BMean (np.ndarray): 1 x K array. 
	    BVar (np.ndarray): 1 x K array. 
	Returns:
	    BSampels (np.ndarray): nSamples x K array.  
	"""
	K = BMean.size
	lower, upper = 0, 1
	MultNormDist = stats.multivariate_normal(mean=BMean,cov=np.diag(BVar))

	BSamples = MultNormDist.rvs(nSamples)
	BSamples[BSamples<0]=lower
	BSamples[BSamples>1]=upper


	return (BSamples)