'''
	For a given set of SNR parameters and a coding scheme calculate the mean depth error
	
	Parameters that control SNR:
	- sourceExponent: Controls source average power per pixel, pAveSourcePerPixel = 10^sourceExponent
	- ambientExponent: Controls ambient average power per pixel, pAveAmbientPerPixel = 10^ambientExponent
	- exposureTimeBudget: Total exposure time divided across all K coding functions in the coding scheme
	- meanBeta: Average effective albedo set for a pixel
	- dMax: Maximum depth that can be imaged by the coding scheme. Decreasing this value is the equivalent of increasing the repetition frequency and hence increases SNR.

	To calculate the mean depth errors of a coding scheme we evenly discretize the depth range of the coding scheme and 
	calculate the depth error for each depth. The depth error for each depth is calculated by simulating that depth many times
	and adding noise each time and calculating the absolute error. We find that simulating the given depth 5000 times is enough to give
	a stable depth error.

	Parameters that affect runtime of calculation:

	- dSample: Spacing between discretized depths across the depth range. These are the depths we are calculating the depth error for. If you increase dSample then we are sampling depths
	farther appart and hence we are sampling less depths across the depth range. That means that we will have to calculate less depth errors. The fewer depth
	errors that are calculated the less accurate the Mean Depth Error Calculated will be.
	- nMonteCarloSamples: Number of depth errors calculated per depth

	Note: Usually expected depth errors are a bit high at the edges of the depth range (e.g. between 0 and 100 mm and 9900 and 10000 mm for a 10000mm depth range)

	To reproduce Figure 4 of 
	http://openaccess.thecvf.com/content_CVPR_2019/papers/Gutierrez-Barragan_Practical_Coding_Function_Design_for_Time-Of-Flight_Imaging_CVPR_2019_paper.pdf
	you need to run this script with various compinations of sourceExponent and ambientExponent. The range that we used for those paramters in those figures was
	around sourceExponent = [7 to 9], ambientExponent = [6 to 9]
'''

# Python imports
import math
# Library imports
import numpy as np
from scipy import signal
from scipy import special
import matplotlib as mpl
import matplotlib.pyplot as plt
from IPython.core import debugger
breakpoint = debugger.set_trace

# Local imports
import CodingFunctions
import Utils
import Decoding

#################### Set Function Parameters
N = 10000
dSample = 200
nMonteCarloSamples = 5000

#################### Get coding functions with total energy = 1
# (ModFs,DemodFs) = CodingFunctions.GetCosCos(N = N, K = 4)
# (ModFs,DemodFs) = CodingFunctions.GetSqSq(N = N, K = 4)
(ModFs,DemodFs) = CodingFunctions.GetHamK3(N = N)
# (ModFs,DemodFs) = CodingFunctions.GetHamK4(N = N)
(ModFs,DemodFs) = CodingFunctions.GetHamK5(N = N)
# (ModFs,DemodFs) = CodingFunctions.GetMultiFreqCosK5(N = N)
(_, K) = ModFs.shape

#################### Coding Function and Scene Parameters
sourceExponent = 8
ambientExponent = 8
#### Global parameters
speedOfLight = 299792458. * 1000. # mm / sec 
#### Sensor parameters
exposureTimeBudget = 0.1 # Total exposure time for the whole coding scheme
T = exposureTimeBudget/float(K) # Exposure time per coding function pair
readNoise = 20 # Standard deviation in photo-electrons
#### Coding function parameters
dMax = 10000. # maximum depth (in mm)
deltaDepth = dMax / N
fMax = speedOfLight/(2*dMax) # Maximum unambiguous repetition frequency (in Hz)
tauMin = 1./fMax
fSampling = dMax*fMax # Sampling frequency of mod and demod functuion
dt = tauMin/float(N)
pAveSourcePerPixel = np.power(10, sourceExponent) # Source power. Avg number of photons emitted by the light source per second. 
# pAveSourcePerPixel = pAveSource/nPixels # Avg number of photons arriving to each pixel per second. If all light is reflected back.
freq = fMax # Fundamental frequency of modulation and demodulation functions
tau = 1/freq
#### Scene parameters
pAveAmbientPerPixel = np.power(10, ambientExponent) # Ambient light power. Avg number of photons per second due to ambient light sources
# pAveAmbientPerPixel = pAveAmbient/nPixels # Avg # of photons per second arriving to each pixel
meanBeta = 1e-4 # Avg fraction of photons reflected from a scene points back to the detector
#### Camera gain parameter
## The following bound is found by assuming the max brightness value is obtained when demod is 1. 
gamma = 1./(meanBeta*T*(pAveAmbientPerPixel+pAveSourcePerPixel)) # Camera gain. Ensures all values are between 0-1.

#### Set list of depths
depths = np.round(np.arange(0, dMax, dSample))
nDepths = len(depths)
print("True Depths: {}".format(depths))

#################### ToF Simulation
## Set area under the curve of outgoing ModF to the totalEnergy
ModFs = Utils.ScaleMod(ModFs, tau=tauMin, pAveSource=pAveSourcePerPixel)
CorrFs = Utils.GetCorrelationFunctions(ModFs,DemodFs,dt=dt)
## Normalized correlation functions, zero mean, unit variance. We have to transpose so that broadcasting works.
NormCorrFs = (CorrFs.transpose() - np.mean(CorrFs, axis=1)) / np.std(CorrFs, axis=1) 
# Transpose it again so that it has dims NxK
NormCorrFs = NormCorrFs.transpose()

# Calculate brightness values without any noise
BValsNoNoise = Utils.ComputeBrightnessVals(ModFs=ModFs, DemodFs=DemodFs, depths=depths, 
						pAmbient=pAveAmbientPerPixel, beta=meanBeta, T=T, tau=tau, dt=dt, gamma=gamma)
#### Add noise
# Calculate gaussian noise variance
# The first term corresponds to the photon noise whose variance are proportional to the amount of photons arriving at the pixel
# The second term is the read noise (scaled by the gain factor)
noiseVar = BValsNoNoise*gamma + math.pow(readNoise*gamma, 2)

# Create array to store the expected depth errors for each depth 
ExpectedDepthErrors = np.zeros((nDepths,))

##### Add noise to all brightness values
print("Expected Depth Errors for Current Coding Scheme")
for i in range(nDepths):
	## Sample from a clipped multivariate gaussian dist of Mean: BValsNoNoise, Covariance: diag(noiseVa)
	# We use a clipped distribution because of saturation at 0 and at 1.0
	CurrBSamples = Utils.GetClippedBSamples(nMonteCarloSamples, BMean=BValsNoNoise[i,:], BVar=noiseVar[i,:])
	# Calculate depths for all BSamples
	CurrDecodedDepths = Decoding.DecodeXCorr(CurrBSamples,NormCorrFs)
	CurrTrueDepth = depths[i]
	# Calcualte expeted depth error for current depth
	ExpectedDepthErrors[i] = np.mean( np.abs( CurrDecodedDepths - CurrTrueDepth ) )

	print("    Expected Depth Error for depth {} = {}".format(CurrTrueDepth, ExpectedDepthErrors[i]))

MeanExpectedDepthError = np.mean(ExpectedDepthErrors)
print("Mean Expected Depth Error = {}".format(MeanExpectedDepthError))
