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
# - calculating direct simple or ordinary predictions
#
##############################################################

# context: two matrices X1, X2 in dimension 2, one lengthscales vector

library(nestedKriging)
X1 <- matrix(c(1.1, 2.2, 3.3, 4.5, 1.2, 1.3, 8.4, 2.1), 4, 2)
X2 <- matrix(c(1.3, 2.6, 3.1, 4.2, 5.3, 6.7), 3, 2)
lengthScales <- c(1.3, 1.44)
Y <- c(1.3, 2.5, 4.5, 6.7)

covType <- "gauss"
doSimpleKriging <- TRUE

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

# 2A. computing kriging predictions
if (doSimpleKriging) {
  krigingType <- "simple"
  krigingTypeDice <- "SK"
  trendDice <- 0
} else {
  krigingType <- "ordinary"
  krigingTypeDice <- "UK"
  trendDice <- NULL  
}

pred <- getKrigingPrediction(X1, Y, X2, lengthScales, covType, krigingType)

# 2B. checking expected results
library(DiceKriging)
km_DiceK <- DiceKriging::km(formula = ~1, design = X1, response = Y, covtype = covType,
                            coef.trend = trendDice, coef.cov = lengthScales, coef.var = 1.0)
pred_DiceK <- DiceKriging::predict(object = km_DiceK, newdata = X2 , type=krigingTypeDice , checkNames=FALSE)

message("mean corresponds to DiceKriging: ", max(abs(pred$mean-pred_DiceK$mean))<1e-10)

message("kriging variance corresponds to DiceKriging: ", max(abs(pred$unitVariance-pred_DiceK$sd^2))<1e-10)

clusters <- rep(1, nrow(X1))
myoutput <- outputLevel(nestedKrigingPredictions = TRUE, predictionBySubmodel = TRUE)
pred_NestedK <- nestedKriging(X=X1, Y=Y, clusters=clusters, x=X2 , covType=covType, param=lengthScales, sd2=1.0,
                              krigingType=krigingType, numThreads=1, verboseLevel=0, outputLevel = myoutput)

message("mean corresponds to NestedKriging submodel: ", max(abs(pred$mean-pred_NestedK$mean_M))<1e-10)

message("kriging variance corresponds to NestedKriging submodel: ", max(abs(pred$unitVariance-pred_NestedK$sd2_M))<1e-10)

pred$mean
