
# Load the libraries required to run this code
library(DiceKriging)     # Necesary to use the Branin toy function.
library(nestedKriging)
# this demo is directly adapted from a nicely commented code of Clement Chevalier

################################################################################
################################################################################
# (I) This is a toy example of use of the 'nestedKriging' function implemented in C++
#     This function computes our 'approximate' best linear predictor, from a potentially large number of observations.
#     The code below should work for n = 10^5 and even n = 10^6 observations. We did not try larger values.
#     If you double the number of observation, the computation time is multiplied by approx. four.

f <- DiceKriging::branin                # the test function used
d <- 2                                  # Dimension of the input domain. For the Branin function this is 2.

set.seed(8)                             # for repeatability
n <- 10000                              # Number of observations

X <- matrix(runif(n*d) , ncol=d)        # Design of experiments. We generate uniform random points in [0,1]^2
Y <- apply(X=X,MARGIN = 1,FUN=f)        # n-dimensional vector of responses

covType <- "exp"                        # choice of the covariance kernel. Possible choices are "gauss", "exp", "matern3_2", "matern5_2"
param <- c(1,1.6)                       # parameters of the covariance function: a vector of size d
sd2 <- 150                              # variance parameter

N <- 100                                # Number of groups used for prediction
clustering <- kmeans(x = X,centers = N) # Here we build the group using a kmeans algorithm. Other choices are of course possible.
clusters <- clustering$cluster          # cluster is a n-dimensional vectors with the group index of each observation.

q <- 100                                # This is the number of new locations where we would like to predict
x <- matrix( runif(q*d),ncol=d   )      # Here we generate these locations randomly. Of course, any location in [0,1]^2 can be chosen here.

krigingType <- "ordinary"

t1 <- Sys.time() # We measure the computation time required to perform the prediction in all q points
prediction <- nestedKrigingDirect(X=X, Y=Y, clusters=clusters, x=x , covType=covType, param=param, sd2=sd2,
                                              krigingType=krigingType, tagAlgo='NK',
                                              numThreads=16, verboseLevel=0, outputLevel=0, globalOptions = c(0), nugget=0.0)
t2 <- Sys.time()
difftime(t2,t1)

expectedY <- apply(X=x,MARGIN = 1,FUN=f)      # true responses at locations x
message('mean squared prediction error = ', mean((prediction$mean-expectedY)^2))

# for q = 100, n = 10,000 , N = 100, exponential kernel, we get the q predictions in approx. 2.5 seconds with a bicore computer of 2016.

################################################################################
################################################################################
# (II) Toy example of use of the 'looErrors' function inplemented in C++ with multithreading
#      This fonction compute some Leave-One-Out (LOO) errors for q points chosen among the n design points.
#      For large n, it is not possible to use q = n because of a O(nq) storage cost in the algorithm.
#      Hence, when n > 10^5, make sure that you choose q << n.

#      For help on the different inputs of the looErrors function, you can read the part (I) above

set.seed(8)
f <- DiceKriging::branin
d <- 2
n <- 10000
X <- matrix(runif(n*d) , ncol=d)
Y <- apply(X=X,MARGIN = 1,FUN=f)

covType <- "exp"
param <- c(1,1.6)
sd2 <- 150

N <- 100
clusters <- kmeans(x = X,centers = N)$cluster

q <- 100
index <- c(1:q)          # This is the choice of the points (referred to using their index: between 1 and n) for which we compute a LOO error
indices <- rep(0,times=n)
indices[index] <- 1      # we built a vector with a 1 in positions 1,...,100 and 0 otherwise. This vector is an argument of the looErrors function
krigingType <- "ordinary"

t1 <- Sys.time()
looPredictions <- nestedKriging::looErrors(X=X, Y=Y, clusters=clusters, indices=indices , covType=covType, param=param, sd2=sd2,
                                      krigingType=krigingType, tagAlgo='nested Kriging LOO',
                                      numThreads=8, verboseLevel=0, outputLevel=0, globalOptions = c(0), nugget=0.0)
t2 <- Sys.time()
difftime(t2,t1)

looMean     <- looPredictions$mean # here are all the predictions, each based on n-1 observations
looExpected <- looPredictions$LOOexpectedPrediction # here are expected values at prediction points
looErrors   <- looMean-looExpected  # here are all the LOO errors
mean( looErrors^2 ) # this is the mean-squared cross-validation error (based on q LOO errors, not n)
looPredictions$LOOErrors$looErrorNestedKriging #same value

message('the mean-squared cross-validation error: ', mean( looErrors^2 ))


# computing 100 LOO cross-validation errors roughly has the same cost as
# computing 100 predictions with the 'nestedKriging' function.

################################################################################
################################################################################
# (III) To verify that looErrors works correctly, we can use the 'nestedKriging' function with n-1 points
#       and predict at the n^th point
#       Running this code requires to have run the previous code from part (II)


index2 <- 68                       # this can be between 1 and 100 (we have calculated the first 100 LOO errors in (II) )
x <- matrix( X[index2,] , nrow=1)  # this is the point where we predict
Xnew <- X[-index2,]                # this is the design of experiments
Ynew <- Y[-index2]                 # the response at the DoE
nnew <- n - 1                      # number of observations
clusternew <- clusters[-index2]    # groups of each observation

t1 <- Sys.time()
predictionWithRemoved <- nestedKrigingDirect(X=Xnew, Y=Ynew, clusters=clusternew, x=x , covType=covType, param=param, sd2=sd2,
                                krigingType=krigingType, tagAlgo='NK',
                                numThreads=8, verboseLevel=0, outputLevel=0, globalOptions = c(0), nugget=0.0)
t2 <- Sys.time()
difftime(t2,t1)

predAtRemoved <- predictionWithRemoved$mean[1]         # this is our prediction at the removed point
predByLooErrors <- looPredictions$mean[index2]         # this is the prediction obtained from the 'looErrors' function
                                                       # and these two numbers are supposed to match !

if (abs(predAtRemoved-predByLooErrors)<1e-7)
  message('the predictions at the removed point match: ', predAtRemoved, ' = ', predByLooErrors)

################################################################################
################################################################################
# (IV) Stochastic gradient algorithm to estimate covariance parameters
#      Here we run a toy example on a 2d-function where the covariance parameters (2 correlation lengths)
#      are estimated:
#      - (i)  by computing the LOO-MSE on a 2d grid (brute-force optimization on a grid)
#      - (ii) using a stochastic gradient algorithm
#
#      For this toy example, we use a 'small' number n of design points and compute all the n LOO cross-validation errors.
#      Toy examples with larger 'n' are of course possible, but take more computation time.
#
#      The goal here is to show that the stochastic gradient algorithm obtains roughly the same result
#      as the brute force optimization.
#      Also, this code suggest some default parameters for the stochastic gradient algorithm
#      (arguments alpha,gamma,a,A,c)

# Toy function (morris)
morris <- function(X) {
  d <- dim(X)[2]
  y <- X[,1]
  for (i in 1:d) {
    y <- y + 2*( (i/2 + X[,i] )/(1+i/2 + X[,i])  - 0.5)
  }
  y <- y + X[,1]*X[,2]
  y
}

seed <- 8
set.seed(seed)                      # For repeatability
d <- 2                              # Number of input variables.

n <- 200                            # Number of observations. To get a fast example, we keep this one low.
X <- matrix(runif(n*d) , ncol=d)    # Design of experiments
Y <- morris(X)                      # Responses

covType <- "exp"                    # Choice of the parametric family of covariance function

N <- 10                             # Choice of the number of groups
clustering <- kmeans(x = X,centers = N) # Construction of the groups. Other choices are of course possible
clusters <- clustering$cluster

krigingType <- "ordinary"           # choice between simple or ordinary Kriging

################################################################
# (Step 1) we try to brute force the estimation of the covariance parameters
# by minimizing the LOO cross validation error on a 2d-grid

indices <- rep(1,times=n)  # we will compute ALL cross validation errors ...
q <- n                 # ... so q = n
nopt <- 50             # this is the resolution of the 2d grid where the brute-force optimization is performed.
sd2 <- 150             # This can be set to any value. The stochastic gradient algorithm will estimate the variance parameter.

subdivisions <- exp(seq(from=-5,length=nopt,to=5))
grid <- expand.grid(subdivisions,subdivisions)

# A total of nopt^2 calls to fast_LOOerror is performed.
# Each call corresponds to different covariance parameters

errorsOnGrid <- rep(0,times = nopt^2)
t1 <- Sys.time()
for(i in 1:nrow(grid)){

  looPredictions <- looErrorsDirect(X=X, Y=Y, clusters=clusters, indices=indices , covType=covType, param=as.numeric(grid[i,]), sd2=sd2,
                                        krigingType=krigingType, numThreads=4, verboseLevel=0, outputLevel=0, nugget=0.0)

  errorsOnGrid[i] <- looPredictions$looError;
}
t2 <- Sys.time()
difftime(t2,t1)
# note that for small n, the algo does not really takes advantage of the parallelism
# the speed increases occurs for larger values of n (... still working on it)

optimalIndex <- which.min(errorsOnGrid)
opimalParamOnGrid <- grid[optimalIndex,]
opimalParamOnGrid     # these are the estimated covariance parameters : approx ~ (1.666 , 3.072) for n=200, seed=8

# and this is the obtained mean squared LOO cross validation error. ~ 0.00014324, for seed=8
errorsOnGrid[optimalIndex]
optimalErrorOnGrig <- errorsOnGrid[optimalIndex]
# note that without nugget the LOO cross validation error does not depend on 'sd2'. So
# this parameter is generally estimated in a next step.

################################################################
# (Step 2) Let's see if a stochastic gradient optimization algorithm is able to find
# a lower (or close) LOO-MSE

niter <- 10000
N <- 10
q <- 10
verboseLevel <- 10   # to track the progress of the computations...
krigingType <- "ordinary"

linit <- c(80,0.1)   # This is our initial 'guess' of the value of the covariance parameters. On purpose we pick a very bad value.
linf <- c(0.01,0.01) # search domain for the covariance parameters : lower bound
lsup <- c(100,100)   # search domain for the covariance parameters : upper bound

alpha <- 0.602       # see, book bathnagar et al
gamma <- 0.101       # see, book bathnagar et al
a <- 200             # see, book bathnagar et al
A <- 1               # see, book bathnagar et al
c <- 0.1             # see, book bathnagar et al


t1 <- Sys.time()
paramEstimation <-nestedKriging::estimParam(X=X, Y=Y, clusters = clusters, q=q, covType = covType, niter=niter,
                                        paramStart = linit, paramLower = linf, paramUpper = lsup,
                                        sd2=sd2, krigingType=krigingType, seed=seed,
                                        alpha = alpha,gamma = gamma,a = a,A = A,c = c, tagAlgo="estimParam",
                                        numThreadsZones=1,numThreads=4,verboseLevel=10,
                                        globalOptions = c(9), nugget=0.0, method = "NK")
t2 <- Sys.time()
difftime(t2,t1)
optimalParam <- paramEstimation$optimalParam  # this is the 'optimal' parameter at the end of the gradient descent

optimalParam                   # we find ~  (1.793 , 3.354) for seed=8
paramEstimation$bestEncounteredError # this is the best encountered error, but for only few q prediction points

# Compute the LOO errors at this 'optimum' for all points
indices <- rep(1,times=n) # we use here all points to compute the LOO errors
t3 <- Sys.time()
detailedErrorsAtOptimum <-looErrors(X=X, Y=Y, clusters=clusters, indices=indices , covType=covType,
          param=optimalParam, sd2=sd2, krigingType=krigingType, tagAlgo='LOO',
          numThreads=4, verboseLevel=0, outputLevel=0, globalOptions = c(0), nugget=0.0)
t4 <- Sys.time()
difftime(t4,t3)

errorAtOptimum <- detailedErrorsAtOptimum$LOOErrors$looErrorNestedKriging
errorAtOptimum
# and this is the obtained mean squared LOO cross validation error. ~ 0.00014375, for seed=8

# the LOO MSE has same order as the brute force optim (depending on the parameters and the seed)

###########################################################
# It remains to estimate the variance parameter.

optimalVariance <- nestedKriging::estimSigma2(X=X, Y=Y, q=n, clusters = clusters, covType=covType,
                                          krigingType = krigingType, param=optimalParam, numThreads = 8,
                                          nugget = 0.0, method="NK", seed=seed)
# So, in the end, the estimated correlation lenghts are:
optimalParam
# and the estimated variance parameter is:
optimalVariance
