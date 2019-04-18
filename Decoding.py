"""Decoding
Decoding functions for time of flight coding schemes.
"""
#### Python imports

#### Library imports
import numpy as np

#### Local imports

def DecodeXCorr(BMeasurements, NormCorrFs):
	"""DecodeXCorr: Generic decoding algorithm that performs a 1D search on the normalized 
	correlation functions.
	
	Args:
	    BMeasurements (np.ndarray): B x K matrix. B sets of K brightness measurements
	    NormCorrFs (np.ndarray): N x K matrix. Normalized Correlation functions. Zero mean
	    unit variance.
	Returns:
	    np.array: decodedDepths 
	"""
	## Normalize Brightness Measurements functions
	NormBMeasurements = (BMeasurements.transpose() - np.mean(BMeasurements, axis=1)) / np.std(BMeasurements, axis=1)
	## Calculate the cross correlation for every measurement and the maximum one will be the depth
	decodedDepths = np.zeros((NormBMeasurements.shape[1],))
	for i in range(NormBMeasurements.shape[1]):
		decodedDepths[i] = np.argmax(np.dot(NormCorrFs, NormBMeasurements[:,i]), axis=0)

	return decodedDepths