
# rm(list=ls(all=TRUE)) # uncomment to clear the environment

#############################################
#                                           #
#  demoA.R : prediction with few clusters   #
#                                           #
#############################################

#########################################
#### library installation and tests
#### in RStudio go to menu Tools, Install Package..., Install from Package Archive File, nestedKriging_xxx.tar.gz

library(nestedKriging)

versionInfo()

#### eventual tests and details on succesful tests (failures are always detailed), see demoE.R
# tests_run()
# tests_run(showSuccess = TRUE)


#########################################
#### known test function and covariances

library(DiceKriging)
testFunction <- DiceKriging::hartman6
d <- 6
covType <- "matern5_2"
sd2 <- 2.2
param <- c(0.7, 0.9, 1.2, 0.7, 0.8, 0.7)

#########################################
#### common parameters
krigingType <- "simple"
n <- 2000 # number of observations
q <- 100  # number of prediction points

#########################################
#### Context: initial design and clustering
set.seed(1)
X <- matrix(runif(n*d), ncol=d)               # initial design
Y <- apply(X=X, MARGIN = 1, FUN=testFunction) # initial response
x <- matrix(runif(q*d), ncol=d)               # prediction points

realvalues <- apply(x, MARGIN = 1, FUN = testFunction) # real values to be predicted

#########################################
#### DiceKriging algorithm

t1 <- Sys.time()
km_DiceK <- DiceKriging::km(formula = ~1, design = X, response = Y, covtype = covType,
                            coef.trend = 0, coef.cov = param, coef.var = sd2)
pred_DiceK <- DiceKriging::predict(object = km_DiceK, newdata = x , type="SK" , checkNames=FALSE)
t2 <- Sys.time()

duration_DiceK <- difftime(t2, t1, units = "secs")
error_DiceK   <- mean(abs(realvalues - pred_DiceK$mean  ))

message("kriging algo duration = ", duration_DiceK, " secs")
message("mean error Dice Kriging = ", error_DiceK)


#########################################
#### Nested kriging algorithm with ONE unique submodel and ONE thread
threads <- 1
t1 <- Sys.time()
clusters <- rep(1, n)
pred_NestedK <- nestedKriging(X=X, Y=Y, clusters=clusters, x=x , covType=covType, param=param, sd2=sd2,
                              krigingType=krigingType, numThreads=threads, verboseLevel=0)
t2 <- Sys.time()

duration_NestedK <- difftime(t2, t1, units = "secs")
error_NestedK <- mean(abs(realvalues - pred_NestedK$mean))
message("nested algo duration = ", duration_NestedK, " secs")
message("mean error Nested Kriging = ", error_NestedK)

ok <- abs(error_NestedK - error_DiceK)<1e-10
message("both give same result? ", ok)

#########################################
#### Nested kriging with SEVERAL submodels

set.seed(1)
N <- 4 #number of submodels
t1 <- Sys.time()
clusters <- kmeans(X, centers=N)$cluster
pred_NestedKbis <- nestedKriging(X=X, Y=Y, clusters=clusters, x=x , covType=covType, param=param, sd2=sd2,
                                 krigingType=krigingType, numThreads=N, verboseLevel=0)
t2 <- Sys.time()

duration_NestedKbis <- difftime(t2, t1, units = "secs")
error_NestedKbis <- mean(abs(realvalues - pred_NestedKbis$mean))
message("nested algo duration = ", duration_NestedKbis, " secs")
message("mean error Nested Kriging = ", error_NestedKbis)

#########################################
#### Precision and speed summary

# Acceleration compared to DiceKriging. 1 cluster => same calculations, same thread number.
speedRatio1 <- as.numeric(duration_DiceK,units="secs")/as.numeric(duration_NestedK, units="secs")
# Acceleration compared to DiceKriging. Approx using N clusters and N threads, faster but approx model
speedRatio2 <- as.numeric(duration_DiceK,units="secs")/as.numeric(duration_NestedKbis, units="secs")
# Acceletation of nestedK using N clusters instead of 1
gainUsingClusters <- speedRatio2/speedRatio1

message("with N = ", N)
message("speed Ratio nestedK(1 cluster) vs DiceK = ", speedRatio1)
message("speed Ratio nestedK(N clusters) vs DiceK = ", speedRatio2)
message("speed Ratio nestedK(N clusters) vs nestedK(1 cluster)= ", gainUsingClusters)

message("mean error DiceK = ", error_DiceK)
message("mean error NestedK(1 cluster) = ", error_NestedK)
message("mean error NestedK(N clusters) = ", error_NestedKbis)
message("with N clusters, even if N is small, the error is increased, but the computational time is largely reduced")
