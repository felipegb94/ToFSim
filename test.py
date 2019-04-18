
import CodingFunctions
import UtilsPlot

N = 1000
K = 4
# (ModFs,DemodFs) = CodingFunctions.GetCosCos(N = N, K = K)
(ModFs,DemodFs) = CodingFunctions.GetHamK3(N = N)
# (ModFs,DemodFs) = CodingFunctions.GetHamK4(N = N)
# (ModFs,DemodFs) = CodingFunctions.GetHamK5(N = N)
(ModFs,DemodFs) = CodingFunctions.GetMultiFreqCosK5(N = N)


UtilsPlot.PlotCodingScheme(ModFs,DemodFs)



