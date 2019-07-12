
import CodingFunctions
from UtilsPlot import *
from Utils import *

N = 1000
K = 3
# (ModFs,DemodFs) = CodingFunctions.GetCosCos(N = N, K = K)
(ModFs,DemodFs) = CodingFunctions.GetSqSq(N = N, K = K)
# (ModFs,DemodFs) = CodingFunctions.GetHamK3(N = N)
# (ModFs,DemodFs) = CodingFunctions.GetHamK4(N = N)
# (ModFs,DemodFs) = CodingFunctions.GetHamK5(N = N)
# (ModFs,DemodFs) = CodingFunctions.GetMultiFreqCosK5(N = N)

plt.clf()

CorrFs = GetCorrelationFunctions(ModFs,DemodFs)
NormCorrFs = NormalizeBrightnessVals(CorrFs)

# PlotCodingScheme(ModFs,DemodFs)

plt.plot(NormCorrFs)



