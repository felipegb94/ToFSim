#### Python imports
#### Library imports
import numpy as np
from scipy import stats
from IPython.core import debugger
breakpoint = debugger.set_trace



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

def cconv( f1, f2 ):
	"""Circular convolution: Calculate the circular convolution for vectors f1 and f2. f1 and f2 are the same size
	
	Args:
		f1 (numpy.ndarray): Nx1 vector	
		f2 (numpy.ndarray): Nx1 vector	
	Returns:
	    f1convf2 (numpy.ndarray): convolution result. N x 1 vector.
	"""
	f1convf2 = np.fft.ifft( np.fft.fft(f1) * np.fft.fft(f2) )
	return f1convf2


def NormalizeBrightnessVals(b_vals):
	"""
		b_vals = n x k numpy matrix where each row corresponds to a set of k brightness measurements		
		Note on implementation: The following implementation is faster than reshaping the mean/stdev data structures. 
	"""
	## Normalized correlation functions, zero mean, unit variance. We have to transpose so that broadcasting works.
	normb_vals = (b_vals.transpose() - np.mean(b_vals, axis=1)) / np.std(b_vals, axis=1) 
	# Transpose it again so that it has dims NxK
	normb_vals = normb_vals.transpose()

	return normb_vals


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
	BVals = BVals[depths.astype(int),:]

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
 

SmoothingWindows = ['flat', 'impulse', 'hanning', 'hamming', 'bartlett', 'blackman']  

def Smooth(x,window_len=11,window='flat'):
    """smooth the data using a window with requested size.
     
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
     
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.
     
    output:
        the smoothed signal
         
    example:
     
    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
     
    see also: 
     
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
     
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """
    #### Validate Inputs
    if(x.ndim != 1):
        raise ValueError("smooth only accepts 1 dimension arrays.")
    if(x.size < window_len):
        raise ValueError("Input vector needs to be bigger than window size.")
    if(window_len<3):
        return x
    if(not window in SmoothingWindows):
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")
    w = GetSmoothingWindow(N=len(x),window=window,window_len=window_len)
    y=np.real(np.fft.ifft(np.fft.fft(x)*np.fft.fft(w)))/(w.sum())
    #### The line below performs the same operation as the line above but slower
    # np.convolve(w/(w.sum()),s,mode='valid')
    return y
 
def GetSmoothingWindow(N=100,window_len=11,window='flat'):
    """smooth the data using a window with requested size.
 
    """
    #### Validate Inputs
    if(N < window_len):
        raise ValueError("Input vector needs to be bigger than window size.")
    if(not window in SmoothingWindows):
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")
    w = np.zeros((N,))
    if window == 'flat': #moving average
        w[0:int(window_len)]=np.ones(int(window_len),'d')
    elif window == 'impulse':
        w[0] = 1 
    else:
        w[0:int(window_len)]=eval('np.'+window+'(int(window_len))')
    return (w / (w.sum()))


def SmoothCodes(modfs, demodfs, window_duty=0.15):
	(N,K) = modfs.shape
	smoothed_modfs = np.zeros((N,K))
	smoothed_demodfs = np.zeros((N,K))
	#### Smooth functions. No smoothing is applied by default
	for i in range(0,K):
		smoothed_modfs[:,i] = Smooth(modfs[:,i],window_len=N*window_duty,window='hanning') 
		smoothed_demodfs[:,i] = Smooth(demodfs[:,i],window_len=N*window_duty,window='hanning')
	return (smoothed_modfs, smoothed_demodfs)