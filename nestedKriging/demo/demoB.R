
#############################################
#                                           #
#  demoB.R : demo with optimization         #
#                                           #
#############################################



rm(list=ls(all=TRUE))

library(nestedKriging)
library(DiceKriging)
library(randtoolbox)


############## global parameters for optimization

testFunction <- hartman6
d <- 6
covType <- "matern5_2"
krigingType <- "simple"
numThreads <- 4

############### rough estimation of covariance parameters
n0Estim <- 500                 # design point size for parameter estimation
X <- sobol(n=n0Estim, dim=d)   # initial input design for covariance estimation
Y <- apply(X=X, MARGIN = 1, FUN=testFunction)  # initial response
m <- km(design=X, response = Y, covtype= covType) # DiceKriging model
param <- coef(m, "range")      # estimation of lengthscales
sd2 <- coef(m, "sd2")          # estimation of variance

############### function for computing expected improvments
# m:    mean prediction
# sd2:  variance prediction
# fmin: current minimum

EI <- function(m, sd2, fmin) {
  s <- sqrt(sd2)
  mbar <- fmin -m
  mbarstd <- mbar/s
  return(mbar*pnorm(mbarstd)+s*dnorm(mbarstd)+1e-99)
}

############### naive EGO algorithm (improved random search)
# at each step: predict at q points, choose p points with proba proportional to EI

#  q:    number of candidates at each step
#  p:    number of chosen candidates at each step (must be <q)
# nstep: number of steps
# n0:    initial design size
# seed:  initializer for random values

naive_EGO <- function(q, p, nstep, n0=1, seed=1)  {
 set.seed(seed)
 p <- min(p,q)
 X <- matrix(runif(n0*d), ncol=d)               # initial design
 Y <- apply(X=X, MARGIN = 1, FUN=testFunction)  # initial response
 fmin <- min(Y)
 minVector <- fmin

 nestedKriging::setNumThreadsBLAS(1)

 for(step in 1:nstep) {
    x <- matrix(runif(q*d), ncol=d)             # new candidates
    n <- nrow(X)                                # current number of observations
    N <- floor(sqrt(n))                         # number of clusters
    cluster <- floor(N*runif(n))                # random clusters

    prediction <- nestedKrigingDirect(X=X, Y=Y, clusters=cluster, x=x , covType=covType, param=param, sd2=sd2,
                                krigingType=krigingType, tagAlgo='', numThreadsZones=1,
                                numThreads=numThreads, verboseLevel=0, outputLevel=0)

    Improvments <- EI(prediction$mean, prediction$sd2, fmin)    # vector of expected improvements
    weightsEI <- Improvments/sum(Improvments)                   # weights used for sampling new data

    ################
    chosenIndexes <- sample(x = 1:q, 1, replace = FALSE, prob = weightsEI) # chosen indexes
    chosenx <- x[chosenIndexes,]      # chosen new data
    radius <- 0.1
    chosenx <- chosenx + matrix(runif(p*d, -radius, radius), ncol=d)

    chosenY <- apply(X=chosenx, MARGIN = 1, FUN=testFunction) # compute new responses
    Y <- c(Y, chosenY)                           # update response values
    X <- rbind(X, chosenx)                       # update input design
    fmin <- min(fmin, chosenY)                   # update current minimum
    minVector <- c(minVector, fmin)
    message('step=', step, ', fmin=', fmin)
 }
 pairs(chosenx)
 message('ended with a design of ', nrow(X), ' points')
 return(minVector)
}

############### Launch basic E

minVector <- naive_EGO(q=200, p=20, nstep=100, seed=1, n0=1)

plot(minVector)

# the algorithm here do not aim at being the more efficient!
# it shows that an optimization algorithm can use a large number of points at each step

