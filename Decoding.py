"""Decoding
Decoding functions for time of flight coding schemes.
"""
#### Python imports

#### Library imports
import numpy as np
from IPython.core import debugger
breakpoint = debugger.set_trace

#### Local imports
import Utils

def DecodeXCorr(b_measurements, norm_corrfs):
	"""DecodeXCorr: Generic decoding algorithm that performs a 1D search on the normalized 
	correlation functions.
	
	Args:
	    b_measurements (np.ndarray): B x K matrix. B sets of K brightness measurements
	    norm_corrfs (np.ndarray): N x K matrix. Normalized Correlation functions. Zero mean
	    unit variance.
	Returns:
	    np.array: decoded_depths 
	"""
	(n_measurements, k) = b_measurements.shape
	## Normalize Brightness Measurements functions
	norm_b_measurements = Utils.NormalizeBrightnessVals(b_measurements)
	## Calculate the cross correlation for every measurement and the maximum one will be the depth
	decoded_depths = np.argmax(np.dot(norm_corrfs, norm_b_measurements.transpose()), axis=0)

	return decoded_depths