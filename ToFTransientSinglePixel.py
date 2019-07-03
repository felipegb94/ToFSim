# Python imports
import math
# Library imports
import numpy as np
from skimage.io import imread 
from scipy import signal, linalg 
from IPython.core import debugger
breakpoint = debugger.set_trace
# Local imports
import CodingFunctions
import Utils
import Decoding


c = 299792458. * 1000. # mm / sec 

#################### Load transient pixel data
dr = 5 # Time resolution of transient renderer in milimeters
dt = 5 / c # Time resolution of transient renderer in milimeters
pixel_id = 100
file_path = '/home/user/repos/ece901/project/hdr_bedroom3_view_6/'
file_name = 'img_0090.hdr'
hdr_data = imread(file_path + file_name)
(n_pixels, n_timebins, _) = hdr_data.shape
impulse_response = np.asarray(hdr_data[pixel_id, :, 1])
#################### Set Coding Function Parameters
freq = 10e6 # 10 mhz
tau = 1 / freq # in seconds
N = round(tau / dt)
max_depth = c*tau / 2.
depths = np.linspace(0, max_depth, N)
#################### Get coding functions with average power = 1
K = 3
# (modfs,demodfs) = CodingFunctions.GetCosCos(N = N, K = K)
(emitted_modfs,emitted_demodfs) = CodingFunctions.GetSqSq(N = N, K = K)
# (modfs,demodfs) = CodingFunctions.GetHamK3(N = N)
# (modfs,demodfs) = CodingFunctions.GetHamK4(N = N)
# (modfs,demodfs) = CodingFunctions.GetHamK5(N = N)
# (modfs,demodfs) = CodingFunctions.GetMultiFreqCosK5(N = N)

(modfs,demodfs) = Utils.SmoothCodes(emitted_modfs,emitted_demodfs, window_duty=0.25)

#################### Get correlation function

#################### Get steady state response through circular convolution
#### make impulse response and coding functions the same size
if(n_timebins < N): 
    impulse_response = np.pad(impulse_response, (0, N-n_timebins), mode='constant', constant_values=(0,0))
    true_n = N
else: 
    n_periods = math.ceil(n_timebins / N)
    true_n = n_periods*N 
    modfs = np.pad(modfs, ((0, true_n-N),(0,0)), mode = 'wrap')
    demodfs = np.pad(demodfs, ((0, true_n-N),(0,0)), mode = 'wrap')
    impulse_response = np.pad(impulse_response, (0, true_n-n_timebins), mode='constant', constant_values=(0,0))


#### Get direct component only.
direct_idx = np.argmax(impulse_response)
direct_impulse_response = np.zeros((true_n,))
direct_impulse_response[direct_idx] = impulse_response[direct_idx]



# true_n = 7093
# true_n = 2998
modfs_response = np.zeros((true_n,K))
modfs_direct_response = np.zeros((true_n,K))
for i in range(K):
    modfs_response[:,i] = Utils.cconv(modfs[:,i], impulse_response)
    modfs_direct_response[:,i] = Utils.cconv(modfs[:,i], direct_impulse_response)
    # M = linalg.circulant(modfs[:,i])
    # modfs_response[:,i] = np.dot(M, impulse_response)
    # modfs_direct_response[:,i] = np.dot(M, direct_impulse_response)

    # breakpoint()
from UtilsPlot import *
plt.clf()
# plt.plot(impulse_response*75, color='#E24A33')
# plt.plot(emitted_modfs[:,0], color='#348ABD')
# plt.plot(modfs[:,0], color='#988ED5')
# plt.plot(modfs_direct_response[:,0]*20 + 0.25, color='#8EBA42')


# plt.plot(modfs_response[:,0]*8  + 0.3, color='#FBC15E')


plt.plot(emitted_demodfs[:,0], color='#777777')
plt.plot(demodfs[:,0], color='#E24A33')



# Out[20]: ['#E24A33', '#348ABD', '#988ED5', '#777777', '#FBC15E', '#8EBA42', '#FFB5B8']

#################### Coding Function and Scene Parameters
# sourceExponent = 9
# ambientExponent = 6
# #### Global parameters
# #### Sensor parameters
# T = 0.1 # Integration time. Exposure time in seconds
# readNoise = 20 # Standard deviation in photo-electrons
# #### Coding function parameters
# dMax = 10000 # maximum depth
# fMax = speedOfLight/(2*float(dMax)) # Maximum unambiguous repetition frequency (in Hz)
# tauMin = 1./fMax
# fSampling = float(dMax)*fMax # Sampling frequency of mod and demod functuion
# dt = tauMin/float(N)
# pAveSourcePerPixel = np.power(10, sourceExponent) # Source power. Avg number of photons emitted by the light source per second. 
# # pAveSourcePerPixel = pAveSource/nPixels # Avg number of photons arriving to each pixel per second. If all light is reflected back.
# freq = fMax # Fundamental frequency of modulation and demodulation functions
# tau = 1/freq
# #### Scene parameters
# pAveAmbientPerPixel = np.power(10, ambientExponent) # Ambient light power. Avg number of photons per second due to ambient light sources
# # pAveAmbientPerPixel = pAveAmbient/nPixels # Avg # of photons per second arriving to each pixel
# meanBeta = 1e-4 # Avg fraction of photons reflected from a scene points back to the detector
# #### Camera gain parameter
# ## The following bound is found by assuming the max brightness value is obtained when demod is 1. 
# gamma = 1./(meanBeta*T*(pAveAmbientPerPixel+pAveSourcePerPixel)) # Camera gain. Ensures all values are between 0-1.

# #### Set list of depths/depth map
# depths = np.round(np.random.rand(5)*dMax).astype(int)
# # depths = np.arange(0,10000)
# print("True Depths: {}".format(depths))

# #################### Simulation
# ## Set area under the curve of outgoing ModF to the totalEnergy
# ModFs = Utils.ScaleMod(modfs, tau=tauMin, pAveSource=pAveSourcePerPixel)
# CorrFs = Utils.GetCorrelationFunctions(modfs,demodfs,dt=dt)
# ## Normalized correlation functions, zero mean, unit variance. We have to transpose so that broadcasting works.
# NormCorrFs = (CorrFs.transpose() - np.mean(CorrFs, axis=1)) / np.std(CorrFs, axis=1) 
# # Transpose it again so that it has dims NxK
# NormCorrFs = NormCorrFs.transpose()

# BVals = Utils.ComputeBrightnessVals(ModFs=modfs, demodfs=demodfs, depths=depths, 
# 						pAmbient=pAveAmbientPerPixel, beta=meanBeta, T=T, tau=tau, dt=dt, gamma=gamma)
# #### Add noise
# # caluclate variance
# noiseVar = BVals*gamma + math.pow(readNoise*gamma, 2) 
# ##### Add noise to all brightness values
# for i in range(depths.size):
#     BVals[i,:] = Utils.GetClippedBSamples(nSamples=1,BMean=BVals[i,:],BVar=noiseVar[i,:])

# decodedDepths = Decoding.DecodeXCorr(BVals,NormCorrFs)

# print("Decoded depths: {},".format(decodedDepths))
