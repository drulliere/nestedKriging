
#############################################
#                                           #
#  demoJ.R : demo with LOO predictions      #
#                                           #
#############################################

#rm(list=ls(all=TRUE))

library(nestedKriging)
library(DiceKriging)

################### parameter description
# testFunction = function that is used (hartman6 is part of DiceKriging package)
# d = dimension
# n = number of observations
# q = number of prediction points
# N = number of sub-models
# krigingType = "simple": Simple Kriging, "ordinary": Ordinary Kriging (first Layer)
# covType = kernel family ( gauss, exp, matern3_2, matern5_2...)
# sd2 = variance of the random field
# param = lengthscale parameters
# verboseLevel = number of steps in progression messages, 0=no messages at all
# numThreads = number of threads for subgroups. Recommended = number of logical cores
# outputLevel = 0: store pred mean & variance in the output, 2:also store large cov matrices between submodels

testFunction <- DiceKriging::hartman6
d <- 6
n <- 20000
q <- 100
N <- 20 #20 => 0.0068...
krigingType <- "simple"
covType <- "matern5_2"

sd2 <- 2.2
param <- c(0.7, 0.9, 1.2, 0.7, 0.8, 0.7)

#################################################
# Context: initial design and clustering

set.seed(1)
X <- matrix(runif(n*d), ncol=d)               # initial design
Y <- apply(X=X, MARGIN = 1, FUN=testFunction) # initial response
x <- matrix(runif(q*d), ncol=d)               # prediction points
clusters <- kmeans(X, centers=N, iter.max=30) # clustering of X into N groups

#################################################
# specific LOO setting

selecProba <- q/n
indices <- rep(0, n)
selectedPredictions <- (runif(n)<selecProba)
indices[selectedPredictions]  <- 1
q <- sum(indices)
x <- X[selectedPredictions,]

#################################################
# Nested kriging algorithm

t1 <- Sys.time()
prediction <- looErrors(X=X, Y=Y, clusters=clusters$cluster, indices=indices , covType=covType, param=param, sd2=sd2,
                        krigingType=krigingType, tagAlgo='demo J',
                        numThreads=4, verboseLevel=8, outputLevel=-3, globalOptions = c(0), nugget=0.0, numThreadsZones = 1, method = "NK")

t2 <- Sys.time()
duration_Nested <- difftime(t2, t1, units = "secs")

#################################################
# average prediction error

realvalues <- apply(x, MARGIN = 1, FUN = testFunction) #real values to be predicted
pred_errors <- abs(realvalues - prediction$mean)
mean_error_Nested <- mean(pred_errors)

message("mean error Nested Kriging = ", mean_error_Nested)
message("nested algo duration = ", duration_Nested, " secs")

