
import CodingFunctions
import UtilsPlot

N = 1000
K = 4
(ModFs,DemodFs) = CodingFunctions.GetCosCos(N = N, K = K)


UtilsPlot.PlotCodingScheme(ModFs,DemodFs)



