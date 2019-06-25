import numpy as np

import CodingFunctions
from UtilsPlot import *
import Utils

N = 10000
K = 4
# (modfs,demodfs) = CodingFunctions.GetCosCos(N = N, K = K)
# (modfs,demodfs) = CodingFunctions.GetHamK3(N = N)
# (modfs,demodfs) = CodingFunctions.GetHamK4(N = N)
# (modfs,demodfs) = CodingFunctions.GetHamK5(N = N)
# (modfs,demodfs) = CodingFunctions.GetMultiFreqCosK5(N = N)
(modfs, demodfs) = CodingFunctions.GetMultiFreqCos( N = N, freq_factors=[1,1,2], phase_shifts=[0,np.pi/2, 0], two_bucket = True  )

corrfs = Utils.GetCorrelationFunctions(modfs,demodfs)
norm_corrfs = Utils.NormalizeBrightnessVals(corrfs)

plt.clf()
# UtilsPlot.PlotCodingScheme(modfs,demodfs)


plt.plot(norm_corrfs)




