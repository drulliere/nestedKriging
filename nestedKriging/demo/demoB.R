
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
X <- matrix(runif(n0Estim*d), ncol=d) 
Y <- apply(X=X, MARGIN = 1, FUN=testFunction)  # initial response
krigingModel <- km(design=X, response = Y, covtype= covType) # DiceKriging model
param <- coef(krigingModel, "range")      # estimation of lengthscales
sd2 <- coef(krigingModel, "sd2")          # estimation of variance

############### function for computing expected improvments
# m:    mean prediction
# sd2:  variance prediction
# fmin: current minimum

EI <- function(m, sd2, fmin) {
  s <- sqrt(sd2)
  mbar <- fmin -m
  mbarstd <- mbar/(s+1e-50)
  return(mbar*pnorm(mbarstd)+s*dnorm(mbarstd)+1e-99)
}

############### naive EGO algorithm (improved random search)
# at each step: predict at q points, choose p points with proba proportional to EI

#  q:    number of candidates at each step
#  p:    number of chosen candidates at each step (must be <q)
# nstep: number of steps
# n0:    initial design size
# seed:  initializer for random values

naive_EGO <- function(q, p, nstep, n0=1, seed=1, pickProportional=TRUE)  {
  set.seed(seed)
  p <- min(p,q)
  X <- matrix(runif(n0*d), ncol=d)               # initial design
  Y <- apply(X=X, MARGIN = 1, FUN=testFunction)  # initial response
  fmin <- min(Y)                                # current minimum
  minVector <- fmin   #all minima, for all steps, will be stored in minVector
  
  nestedKriging::setNumThreadsBLAS(1)  #avoid using parallel computing inside linear algebra tools
  
  for(step in 1:nstep) {
    xnew <- matrix(runif(q*d), ncol=d)          # q new design points
    n <- nrow(X)                                # current number of observations
    N <- floor(sqrt(n))                         # number of clusters
    clusters <- floor(N*runif(n))               # random clusters: cluster number by point
    
    prediction <- nestedKrigingDirect(X=X, Y=Y, clusters=clusters, x=xnew, covType=covType, param=param, sd2=sd2,
                                      krigingType=krigingType, numThreads=numThreads, verboseLevel=-1, outputLevel=0)
    
    improvments <- EI(prediction$mean, prediction$sd2, fmin)    # vector of expected improvements
    weightsEI <- improvments/sum(improvments)                   # weights used for sampling new data
    
    if (pickProportional) {       
        # pick p points chosen with proba propotional to EI: 
        chosenIndexes <- sample(x = 1:q, size=p, replace = FALSE, prob = weightsEI) # chosen indexes
     } else {                      
        # pick the p points having the best EI:
        chosenIndexes <- order(weightsEI, decreasing = TRUE)[1:p]
    }
    chosenxnew <- matrix(xnew[chosenIndexes,], nrow=p)           # chosen new design points
    chosenY <- apply(X=chosenxnew, MARGIN = 1, FUN=testFunction) # compute new responses
    Y <- c(Y, chosenY)                           # update response values
    X <- rbind(X, chosenxnew)                    # update input design
    fmin <- min(fmin, chosenY)                   # update current minimum
    minVector <- c(minVector, fmin)              # update the vector of all minima
    message('step=', step, ', fmin=', fmin)      # show step and current minimum
  }
  pairs(chosenxnew)   # show explored location, to see if it is very focused or widespread
  message('ended with a design of ', nrow(X), ' points')
  return(minVector)
}

############### Launch basic EGO

minVector <- naive_EGO(q=100, p=20, nstep=200, seed=1, n0=1, pickProportional=FALSE)

plot(minVector) # plot the vector of current minima by step

# the algorithm here do not aim at being the more efficient!
# it shows that an optimization algorithm can use a large number of points at each step

