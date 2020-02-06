##############################################################
#                                                            #
#  demoM.R : demo with functions giving intermediate outputs #
#                                                            #
##############################################################

# this demo shows side results: 
# - not directly related to the nestedKriging method
# - not optimized for a direct use outside nestedKriging
#
# for example:
# - calculating correlations or cross-correlations
#
##############################################################

# context: two matrices X1, X2 in dimension 2, one lengthscales vector

library(nestedKriging)
X1 <- matrix(c(1.1, 2.2, 3.3, 4.5), 2, 2)
X2 <- matrix(c(1.3, 2.6, 3.1, 4.2, 5.3, 6.7), 3, 2)
lengthScales <- c(1.3, 1.44)
covType <- "gauss"

# 1A. computing correlations or cross correlations for matrices X1, X2

K1 <- getCorrMatrix(X1, lengthScales, covType)
K12 <- getCrossCorrMatrix(X1, X2, lengthScales, covType)
message("K1[1,2] = ", K1[1,2])

# 1B. testing expected results

K11 <- getCrossCorrMatrix(X1, X1, lengthScales, covType)
squareDistance <- ((X1[1,1]-X1[2,1])/lengthScales[1])^2+((X1[1,2]-X1[2,2])/lengthScales[2])^2
expectedCorrGauss <- exp(-0.5*squareDistance)

message("K1 is equal to K11: ", max(abs(K1-K11))<1e-10)
message("K1 is symmetric: ", abs(K1[1,2]-K1[2,1])<1e-10)
message("K1(1,2) equals gauss correlation: ", abs(K1[1,2]- expectedCorrGauss)<1e-10)
