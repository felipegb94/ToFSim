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
#################### Get coding functions with total energy = 1
# K = 4
# (ModFs,DemodFs) = CodingFunctions.GetCosCos(N = N, K = K)
(ModFs,DemodFs) = CodingFunctions.GetHamK3(N = N)
# (ModFs,DemodFs) = CodingFunctions.GetHamK4(N = N)
# (ModFs,DemodFs) = CodingFunctions.GetHamK5(N = N)
# (ModFs,DemodFs) = CodingFunctions.GetMultiFreqCosK5(N = N)

#################### Coding Function and Scene Parameters
sourceExponent = 9
ambientExponent = 6
#### Global parameters
speedOfLight = 299792458. * 1000. # mm / sec 
#### Sensor parameters
T = 0.1 # Integration time. Exposure time in seconds
readNoise = 20 # Standard deviation in photo-electrons
#### Coding function parameters
dMax = 10000 # maximum depth
fMax = speedOfLight/(2*float(dMax)) # Maximum unambiguous repetition frequency (in Hz)
tauMin = 1./fMax
fSampling = float(dMax)*fMax # Sampling frequency of mod and demod functuion
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

#### Set list of depths/depth map
depths = np.round(np.random.rand(5)*dMax).astype(int)
# depths = np.arange(0,10000)
print("True Depths: {}".format(depths))

#################### Simulation
## Set area under the curve of outgoing ModF to the totalEnergy
ModFs = Utils.ScaleMod(ModFs, tau=tauMin, pAveSource=pAveSourcePerPixel)
CorrFs = Utils.GetCorrelationFunctions(ModFs,DemodFs,dt=dt)
## Normalized correlation functions, zero mean, unit variance. We have to transpose so that broadcasting works.
NormCorrFs = (CorrFs.transpose() - np.mean(CorrFs, axis=1)) / np.std(CorrFs, axis=1) 
# Transpose it again so that it has dims NxK
NormCorrFs = NormCorrFs.transpose()

BVals = Utils.ComputeBrightnessVals(ModFs=ModFs, DemodFs=DemodFs, depths=depths, 
						pAmbient=pAveAmbientPerPixel, beta=meanBeta, T=T, tau=tau, dt=dt, gamma=gamma)
#### Add noise
# caluclate variance
noiseVar = BVals*gamma + math.pow(readNoise*gamma, 2) 
##### Add noise to all brightness values
for i in range(depths.size):
    BVals[i,:] = Utils.GetClippedBSamples(nSamples=1,BMean=BVals[i,:],BVar=noiseVar[i,:])

decodedDepths = Decoding.DecodeXCorr(BVals,NormCorrFs)

print("Decoded depths: {},".format(decodedDepths))
