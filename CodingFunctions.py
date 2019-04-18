#### Python imports
import math

#### Library imports
import numpy as np
import scipy as sp
from scipy import signal
# from IPython.core import debugger
# breakpoint = debugger.set_trace

#### Local imports
import Utils

TotalEnergyDefault = 1.
TauDefault = 1.
AveragePowerDefault = TotalEnergyDefault / TauDefault

def GetCosCos(N=1000, K=3, tau=TauDefault, totalEnergy=TotalEnergyDefault):
	"""GetCosCos: Get modulation and demodulation functions for sinusoid coding scheme. The shift
	between each demod function is 2*pi/k where k can be [3,4,5...]
	
	Args:
	    N (int): N - Number of Samples
	    k (int): k - Number of coding function
		freqFactor (float): Multiplicative factor to the fundamental frequency we want to use.

	Returns:
	    np.array: modFs 
	    np.array: demodFs 
	"""
	#### Allocate modulation and demodulation vectors
	modFs = np.zeros((N,K))
	demodFs = np.zeros((N,K))
	t = np.linspace(0, 2*np.pi, N)
	dt = float(tau) / float(N)
	#### Declare base sin function
	cosF = (0.5*np.cos(t)) + 0.5
	#### Set each mod/demod pair to its base function and scale modulations
	for i in range(0,K):
		## No need to apply phase shift to modF
		modFs[:,i] = cosF
		## Scale  modF so that area matches the total energy
		modFs[:,i] = Utils.ScaleAreaUnderCurve(modFs[:,i], dx=dt, desiredArea=totalEnergy)
		## Apply phase shift to demodF
		demodFs[:,i] = cosF
	#### Apply phase shifts to demodF
	shifts = np.arange(0, K)*(float(N)/float(K))
	demodFs = Utils.ApplyKPhaseShifts(demodFs,shifts)
	#### Return coding scheme
	return (modFs, demodFs)


# def GetHamK3SqSq(N=1000, maxInstantPowerFactor=6., windowDutyCycle = 0):
# 	"""GetHamK3SqSq: Get modulation and demodulation functions for the coding scheme
# 		HamK3 - Sq16Sq50.
	
# 	Args:
# 	    N (int): N
	
# 	Returns:
# 	    modFs: Figure handle
# 	    demodFs: Axis handle
# 	"""
# 	#### Set some parameters
# 	K = 3
# 	dt = float(TauDefault) / float(N)
# 	#### Allocate modulation and demodulation vectors
# 	modFs = np.zeros((N,K))
# 	demodFs = np.zeros((N,K))
# 	#### Prepare modulation functions
# 	modDuty = 1./maxInstantPowerFactor
# 	for i in range(0,K):
# 		modFs[0:math.floor(modDuty*N),i] = maxInstantPowerFactor*AveragePowerDefault
# 	#### Prepare demodulation functions
# 	## Make shape of function
# 	demodDuty = 1./2.
# 	for i in range(0,K):
# 		demodFs[0:math.floor(demodDuty*N),i] = 1.
# 	## Apply necessary phase shift
# 	shifts = [0, (1./3.)*N, (2./3.)*N]
# 	demodFs = UtilsData.ApplyKPhaseShifts(demodFs,shifts)
# 	#### Smooth functions. No smoothing is applied by default
# 	for i in range(0,K):
# 		modFs[:,i] = UtilsData.Smooth(modFs[:,i],window_len=N*windowDutyCycle,window='hanning')	
# 		demodFs[:,i] = UtilsData.Smooth(demodFs[:,i],window_len=N*windowDutyCycle,window='hanning')
# 		## Make sure area under modFs is the total energy 	
# 		modFs[:,i] = UtilsData.ScaleAreaUnderCurve(modFs[:,i], dx=dt, desiredArea=TotalEnergyDefault)

# 	return (modFs, demodFs)


# def GetHamK4SqSq(N=1000, maxInstantPowerFactor=12., windowDutyCycle = 0):
# 	"""GetHamK3Sq8Sq50: Get modulation and demodulation functions for the coding scheme
# 		HamK3 - Sq16Sq50.
	
# 	Args:
# 	    N (int): N
	
# 	Returns:
# 	    modFs: Figure handle
# 	    demodFs: Axis handle
# 	"""
# 	#### Set some parameters
# 	K = 4
# 	dt = float(TauDefault) / float(N)
# 	#### Allocate modulation and demodulation vectors
# 	modFs = np.zeros((N,K))
# 	demodFs = np.zeros((N,K))
# 	#### Prepare modulation functions
# 	modDuty = 1./maxInstantPowerFactor
# 	for i in range(0,K):
# 		modFs[0:math.floor(modDuty*N),i] = maxInstantPowerFactor*AveragePowerDefault
# 	#### Prepare demodulation functions
# 	## Make shape of function
# 	demodDuty1 = np.array([6./12.,6./12.])
# 	shift1 = 5./12.
# 	demodDuty2 = np.array([6./12.,6./12.])
# 	shift2 = 2./12.
# 	demodDuty3 = np.array([3./12.,4./12.,3./12.,2./12.])
# 	shift3 = 0./12.
# 	demodDuty4 = np.array([2./12.,3./12,4./12.,3./12.])
# 	shift4 = 4./12.
# 	shifts = [shift1*N, shift2*N, shift3*N, shift4*N]
# 	demodDutys = [demodDuty1, demodDuty2, demodDuty3, demodDuty4]
# 	for i in range(0,K):
# 		demodDuty = demodDutys[i]
# 		startIndeces = np.floor((np.cumsum(demodDuty) - demodDuty)*N)
# 		endIndeces = startIndeces + np.floor(demodDuty*N) - 1
# 		for j in range(len(demodDuty)):
# 			if((j%2) == 0):
# 				demodFs[int(startIndeces[j]):int(endIndeces[j]),i] = 1.
# 	## Apply necessary phase shift
# 	demodFs = UtilsData.ApplyKPhaseShifts(demodFs,shifts)
# 	#### Smooth functions. No smoothing is applied by default
# 	for i in range(0,K):
# 		modFs[:,i] = UtilsData.Smooth(modFs[:,i],window_len=N*windowDutyCycle,window='hanning')	
# 		demodFs[:,i] = UtilsData.Smooth(demodFs[:,i],window_len=N*windowDutyCycle,window='hanning')	

# 	return (modFs, demodFs)


# def GetHamK5SqSq(N=1000,maxInstantPowerFactor=30., windowDutyCycle = 0):
# 	"""GetHamK3Sq8Sq50: Get modulation and demodulation functions for the coding scheme
# 		HamK3 - Sq16Sq50.
	
# 	Args:
# 	    N (int): N
	
# 	Returns:
# 	    modFs: Figure handle
# 	    demodFs: Axis handle
# 	"""
# 	#### Set some parameters
# 	K = 5
# 	dt = float(TauDefault) / float(N)
# 	#### Allocate modulation and demodulation vectors
# 	modFs = np.zeros((N,K))
# 	demodFs = np.zeros((N,K))
# 	#### Prepare modulation functions
# 	modDuty = 1./maxInstantPowerFactor
# 	for i in range(0,K):
# 		modFs[0:math.floor(modDuty*N),i] = maxInstantPowerFactor*AveragePowerDefault
# 	#### Prepare demodulation functions
# 	## Make shape of function
# 	demodDuty1 = np.array([15./30.,15./30.])
# 	shift1 = 15./30.
# 	demodDuty2 = np.array([15./30.,15./30.])
# 	shift2 = 7./30.
# 	demodDuty3 = np.array([8./30.,8./30.,7./30.,7./30.])
# 	shift3 = 3./30.
# 	demodDuty4 = np.array([4./30.,4./30.,4./30.,4./30.,3./30.,4./30.,4./30.,3./30.])
# 	shift4 = 1./30.
# 	demodDuty5 = np.array([2./30.,2./30.,2./30.,2./30.,2./30.,2./30.,2./30.,
# 							3./30.,2./30.,2./30.,2./30.,2./30.,3./30.,2./30])
# 	shift5 = 4./30.
# 	shifts = [shift1*N, shift2*N, shift3*N, shift4*N, shift5*N]
# 	demodDutys = [demodDuty1, demodDuty2, demodDuty3, demodDuty4, demodDuty5]
# 	for i in range(0,K):
# 		demodDuty = demodDutys[i]
# 		startIndeces = np.floor((np.cumsum(demodDuty) - demodDuty)*N)
# 		endIndeces = startIndeces + np.floor(demodDuty*N) - 1
# 		for j in range(len(demodDuty)):
# 			if((j%2) == 0):
# 				demodFs[int(startIndeces[j]):int(endIndeces[j]),i] = 1.

# 	## Apply necessary phase shift
# 	demodFs = UtilsData.ApplyKPhaseShifts(demodFs,shifts)
# 	#### Smooth functions. No smoothing is applied by default
# 	for i in range(0,K):
# 		modFs[:,i] = UtilsData.Smooth(modFs[:,i],window_len=N*windowDutyCycle,window='hanning')	
# 		demodFs[:,i] = UtilsData.Smooth(demodFs[:,i],window_len=N*windowDutyCycle,window='hanning')	

# 	return (modFs, demodFs)
