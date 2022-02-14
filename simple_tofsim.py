'''
	This script simulates clean ToF data from a depth image. It is very simple and ignores a lot of the scaling factors that each tof measurements
	usually undergoes (e.g., reflectivity, exposure time, etc)
'''

#### Python imports

#### Library imports
import numpy as np
import matplotlib.pyplot as plt
from IPython.core import debugger
breakpoint = debugger.set_trace

#### Local imports


def generate_m_phase_sinusoid(n, m_phase=3, freqs=[1]):
	n_freqs = len(freqs)
	k = m_phase*n_freqs
	corrfs = np.zeros((n, k))
	phase_step = 2*np.pi / m_phase
	domain = np.arange(0, n) / n
	for i in range(n_freqs):
		curr_f = freqs[i]
		for j in range(m_phase):
			curr_k = (i*m_phase) + j
			curr_phi = (j*phase_step) / curr_f
			# print("Curr F = {}, Curr K = {}, Curr Phi = {}".format(curr_f, curr_k, curr_phi))
			curr_sinusoid = 0.5*np.cos((2*np.pi*curr_f*domain) + curr_phi) + 0.5 
			corrfs[:, curr_k] = curr_sinusoid
	return corrfs

def circular_conv( v1, v2, axis=-1 ):
	"""1D Circular convolution: Calculate the circular convolution for vectors v1 and v2. v1 and v2 are the same size
	
	Args:
		v1 (numpy.ndarray): ...xN vector	
		v2 (numpy.ndarray): ...xN vector	
	Returns:
		v1convv2 (numpy.ndarray): convolution result. N x 1 vector.
	"""
	v1convv2 = np.fft.irfft( np.fft.rfft( v1, axis=axis ) * np.fft.rfft( v2, axis=axis ), axis=axis, n=v1.shape[axis] )
	return v1convv2

def circular_corr( v1, v2, axis=-1 ):
	"""1D Circular correlation: Calculate the circular correlation for vectors v1 and v2. v1 and v2 are the same size
	
	Args:
		v1 (numpy.ndarray): Nx1 vector	
		v2 (numpy.ndarray): Nx1 vector	
	Returns:
		v1corrv2 (numpy.ndarray): correlation result. N x 1 vector.
	"""
	v1corrv2 = np.fft.ifft( np.fft.fft( v1, axis=axis ).conj() * np.fft.fft( v2, axis=axis ), axis=axis ).real
	return v1corrv2

def generate_hamk4(n):
	from CodingFunctions import GetHamK4
	(modfs, demodfs) = GetHamK4(n)
	corrfs = circular_corr(modfs, demodfs, axis=0)
	return corrfs

def norm(v, axis=-1):
	# Add 1e-7 to make sure no division by 0
	return v / (np.linalg.norm(v, ord=2, axis=axis, keepdims=True) + 1e-7)

def zero_norm(v, axis=-1):
	return norm(v - v.mean(axis=axis,keepdims=True))

def zncc_depth_est(b_img, corrfs):
	'''
		Assumes that the last dimension of b_img and corrfs match
	'''
	EPSILON = 1e-6
	zn_corrfs = zero_norm(corrfs, axis=-1)
	zn_b_img = zero_norm(b_img)
	zncc_img = np.matmul(zn_corrfs, zn_b_img[..., np.newaxis] ).squeeze(-1)  
	depth_bin_img = np.argmax(zncc_img,axis=-1)
	return depth_bin_img

if __name__=='__main__':

	# 
	n = 1000
	k = 4
	max_depth = 10 # 10 meters
	depth_res = max_depth / n

	# generate corrfs (can sinusoid or hamiltonian)
	# corrfs = generate_hamk4(n)
	corrfs = generate_m_phase_sinusoid(n, m_phase=k, freqs=[1])
	# plot corrfs
	plt.clf()
	plt.plot(corrfs)

	# load depth image (units are in meters)
	depth_img = np.load('./sample_data/cbox_depthmap_nr-240_nc-320.npy')
	(nr, nc) = depth_img.shape

	# simulate ToF measurements
	# NOTE: The code below can be easily vectorized, but I left it as for loops to make it easier to undersnat
	b_img = np.zeros((nr, nc, k))
	# For each pixel, get current depth, and find the K correlation function values associated with that depth
	for i in range(nr):
		for j in range(nc):
			curr_depth = depth_img[i,j]
			curr_depth_bin = np.floor(curr_depth / depth_res)
			b_img[i,j,:] = corrfs[int(curr_depth_bin), :]


	# Estimate depths
	est_depth_img = zncc_depth_est(b_img, corrfs)*depth_res


	plt.clf()
	plt.subplot(4,1,1)
	plt.plot(corrfs)
	plt.title("Correlation Functions")
	plt.subplot(4,4,5)
	plt.imshow(b_img[:,:,0])
	plt.title("Brighness Image [0]")
	plt.subplot(4,4,6)
	plt.imshow(b_img[:,:,1])
	plt.title("Brighness Image [1]")
	plt.subplot(4,4,7)
	plt.imshow(b_img[:,:,2])
	plt.title("Brighness Image [2]")
	plt.subplot(4,4,8)
	plt.imshow(b_img[:,:,3])
	plt.title("Brighness Image [3]")

	plt.subplot(2,3,4)
	plt.imshow(depth_img, vmin=depth_img.min(), vmax=depth_img.max())
	plt.title("True Depths in m")
	plt.colorbar()
	
	plt.subplot(2,3,5)
	plt.imshow(est_depth_img, vmin=depth_img.min(), vmax=depth_img.max())
	plt.title("Estimated Depths in m")
	plt.colorbar()

	plt.subplot(2,3,6)
	plt.imshow(np.abs(est_depth_img-depth_img)*1000, vmin=0, vmax=2*depth_res*1000)
	plt.title("Depth Errors in mm")
	plt.colorbar()

	print("Note that the maximum depth error is bounded by the depth resolution ({})".format(depth_res))


	