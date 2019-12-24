
#############################################
#                                           #
#  demoL.R : demo with benchmark            #
#                                           #
#############################################

rm(list=ls(all=TRUE))

library(nestedKriging)
library(DiceKriging)

library(ggplot2)
library(microbenchmark)

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

# Default parameters
testFunction <- hartman6
d <- 6
n <- 20000
q <- 100
N <- 100
krigingType <- "simple"
#covType <- "matern5_2"

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
# Function to be tested in Benchmark

myfunction <- function(myKernel) {
  nestedKrigingDirect(X=X, Y=Y, clusters=clusters$cluster, x=x , covType=myKernel, param=param, sd2=sd2,
                      krigingType=krigingType, numThreads=16, verboseLevel=0, outputLevel=0)
}

option1 <- "gauss"
option2 <- "gauss.legacy"
option3 <- "gauss.approx"
repeatBenchmark <- 10

################################################ One prediction to check

t1 <- Sys.time()
prediction <- myfunction(myKernel = option1)
realvalues <- apply(x, MARGIN = 1, FUN = testFunction) #real values to be predicted
pred_errors <- abs(realvalues - prediction$mean)
mean_error_Nested <- mean(pred_errors)
t2 <- Sys.time()
message("duration option1=", difftime(t2, t1, units = "secs"))
message("mean error=", mean_error_Nested)

################################################
t1 <- Sys.time()
res <- microbenchmark(myfunction(option1),myfunction(option2),myfunction(option3), times=1)
t2 <- Sys.time()
duration_OneShot <- difftime(t2, t1, units = "secs")
duration_OneShot
estimatedTotalDurationMinutes <- difftime(t2, t1, units = "mins")*repeatBenchmark
message("estimated duration of benchmark = ", estimatedTotalDurationMinutes, " mins")

t1 <- Sys.time()
res <- microbenchmark(myfunction(option1),myfunction(option2),myfunction(option3), times=repeatBenchmark)
plot(res)
ggplot2::autoplot(res, CI=TRUE, pval=TRUE, plotTable=TRUE)
boxplot(res, unit = "t", log = TRUE, horizontal = FALSE)
t2 <- Sys.time()

duration_Benchmark <- difftime(t2, t1, units = "mins")
message("duration of benchmark = ", duration_Benchmark, " mins")

svgres <- res
res
message("option1=", option1, ", option2=", option2, ", option3=", option3)
